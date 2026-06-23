//! Import edits for renaming Python modules and packages.
//!
//! See [`will_rename_paths`] for the supported policy.

use std::sync::Mutex;

use crate::references::contains_identifier;
use ruff_db::files::{File, system_path_to_file};
use ruff_db::source::source_text;
use ruff_db::system::{SystemPath, SystemPathBuf};
use ruff_python_ast::token::{TokenKind, Tokens};
use ruff_python_ast::{
    self as ast, AnyNodeRef,
    visitor::source_order::{SourceOrderVisitor, TraversalSignal},
};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use ty_module_resolver::{
    Module, ModuleName, ModuleResolveMode, file_to_module, resolve_real_module_confident,
    search_paths,
};
use ty_project::Db;
use ty_python_core::{
    FileScopeId, definition::DefinitionKind, place::ScopedPlaceId, semantic_index,
};
use ty_python_semantic::types::Type;
use ty_python_semantic::{
    HasType, ImportAliasResolution, ResolvedDefinition, SemanticModel,
    definitions_for_attribute_with_alias_resolution, definitions_for_imported_symbol,
    definitions_for_name,
};

/// Returns the text replacements that should be applied before renaming Python modules.
///
/// Every supported item in `renames` is handled independently. Python files can move within or
/// between import roots, and directories can contain regular packages, namespace-package
/// portions, or both. When import roots overlap, the first valid path in resolver order determines
/// the destination module name. A move that preserves its module name needs no edits and is
/// ignored. Imports keep their existing statement shape, explicit aliases remain unchanged, and
/// direct module usages are rewritten when semantic analysis resolves them to a moved file. For
/// example, moving `old_pkg/tool.py` to `new_pkg/tool.py` changes:
///
/// ```python
/// from old_pkg import tool
/// ```
///
/// to `from new_pkg import tool`.
///
/// A moved file that contains a relative import is omitted because ty does not rebase relative
/// imports against the file's destination. Directory items use only descendants in `files`; this
/// function does not walk the renamed directory.
/// Unmoved portions of a split namespace are not exhaustively searched for implicit relative
/// references to descendants in the moved portion. An unsupported path or module does not suppress
/// edits for other rename items. If one affected file contains an import or reference that cannot
/// be rewritten with non-overlapping text replacements, edits for that file are omitted while other
/// files remain. A propagated re-export edit is retained only when the edit that changes the
/// defining import is also retained. This includes cross-root moves of unaliased dotted imports and
/// references that cannot retain their existing bound root. A file is also omitted when one binding
/// would be changed by an implicit import but preserved by an explicit alias, when a name target
/// would require a dotted name, or when an attribute path would add or remove module components.
/// When source and stub files resolve to the same module, import rewrites follow the runtime source
/// file; a stub-only move does not redirect imports away from a source that stays behind.
///
/// This is a semantic best-effort operation, not a proof that the destination workspace is valid.
/// It does not preflight destination shadowing, binding collisions, collisions inferred from a
/// moved directory's descendants, or completeness through dynamic references. A consumer with a
/// bare import of an aggregate namespace is omitted because ty cannot know whether another portion
/// will keep the old root importable after one physical directory moves.
///
/// # Arguments
///
/// * `db` - The semantic database used to resolve modules and analyze references.
/// * `renames` - The filesystem path renames reported by the client.
/// * `files` - The candidate Python files that may contain affected imports or references. This
///   also determines which descendants are inspected for a directory item.
/// * `file_is_in_scope` - Returns whether a candidate or renamed file belongs to the request's
///   scope. Files outside the scope are not inspected or edited.
pub fn will_rename_paths(
    db: &dyn Db,
    renames: &[PathRename],
    files: impl IntoIterator<Item = File>,
    file_is_in_scope: impl Fn(File) -> bool,
) -> Vec<FileRenameEdit> {
    let mut files: FxHashSet<_> = files
        .into_iter()
        .filter(|file| file_is_in_scope(*file))
        .collect();
    let plan = RenamePlan::new(db, renames, &files);
    if plan.rules.is_empty() {
        return Vec::new();
    }

    // A file named directly by a file item can be excluded from the project index but still
    // contain an import whose meaning changes at the destination.
    files.extend(
        plan.moved_files
            .iter()
            .copied()
            .filter(|file| file_is_in_scope(*file)),
    );

    let db = Db::dyn_clone(db);
    let collected = Mutex::new(Vec::new());

    let plan_ref = &plan;
    let collected_ref = &collected;
    rayon::scope(move |scope| {
        #[expect(
            clippy::iter_over_hash_type,
            reason = "Rayon task order is unspecified and edits are sorted before returning"
        )]
        for file in files {
            let db = Db::dyn_clone(&*db);
            let plan = plan_ref;
            let collected = collected_ref;
            scope.spawn(move |_| {
                if let Some(result) = rename_edits_for_file(&*db, file, plan)
                    && !result.edits.is_empty()
                {
                    collected
                        .lock()
                        .expect("rename edit worker should not panic while holding the lock")
                        .push((file, result));
                }
            });
        }
    });

    let mut results: FxHashMap<_, _> = collected
        .into_inner()
        .expect("rename edit worker should not panic while holding the lock")
        .into_iter()
        .collect();
    // A re-export edit is valid only while every defining import in its dependency chain also has
    // an edit. Remove failed definitions and their dependents to a fixed point.
    loop {
        let files_with_edits: FxHashSet<_> = results.keys().copied().collect();
        let previous_len = results.len();
        results.retain(|_, result| result.required_edits_from.is_subset(&files_with_edits));
        if results.len() == previous_len {
            break;
        }
    }
    let mut edits: Vec<_> = results
        .into_values()
        .flat_map(|result| result.edits)
        .collect();
    edits.sort_by(|left, right| {
        left.file
            .cmp(&right.file)
            .then_with(|| left.range.start().cmp(&right.range.start()))
            .then_with(|| left.range.end().cmp(&right.range.end()))
            .then_with(|| left.new_text.cmp(&right.new_text))
    });
    edits
}

/// A Python module or directory rename received from the client.
#[derive(Debug, Clone)]
pub struct PathRename {
    /// The current filesystem path.
    old_path: SystemPathBuf,
    /// The requested filesystem path.
    new_path: SystemPathBuf,
    /// Whether the path names a file or directory.
    kind: PathRenameKind,
}

impl PathRename {
    /// Creates a file rename from `old_path` to `new_path`.
    pub fn file(old_path: SystemPathBuf, new_path: SystemPathBuf) -> Self {
        Self {
            old_path,
            new_path,
            kind: PathRenameKind::File,
        }
    }

    /// Creates a directory rename from `old_path` to `new_path`.
    pub fn directory(old_path: SystemPathBuf, new_path: SystemPathBuf) -> Self {
        Self {
            old_path,
            new_path,
            kind: PathRenameKind::Directory,
        }
    }
}

/// A text replacement to apply before renaming a Python module or package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRenameEdit {
    /// The file to edit before the rename.
    file: File,
    /// The source range to replace.
    range: TextRange,
    /// The replacement text.
    new_text: String,
}

impl FileRenameEdit {
    /// Returns the file, source range, and replacement text that make up this edit.
    pub fn into_parts(self) -> (File, TextRange, String) {
        (self.file, self.range, self.new_text)
    }
}

/// Whether a path identifies a Python file or a directory containing Python modules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PathRenameKind {
    File,
    Directory,
}

/// A batch of non-conflicting module rename rules.
struct RenamePlan {
    rules: Vec<RenameRule>,
    moved_files: FxHashSet<File>,
}

impl RenamePlan {
    fn new(db: &dyn Db, renames: &[PathRename], files: &FxHashSet<File>) -> Self {
        let items: Vec<_> = renames
            .iter()
            .filter_map(|rename| ResolvedRename::from_path_rename(db, rename, files))
            .collect();
        let mut conflicting = FxHashSet::default();

        for left in 0..items.len() {
            for right in left + 1..items.len() {
                if items[left].rule.conflicts_with(db, &items[right].rule) {
                    conflicting.insert(left);
                    conflicting.insert(right);
                }
            }
        }

        let mut plan = Self {
            rules: Vec::new(),
            moved_files: FxHashSet::default(),
        };
        for (index, item) in items.into_iter().enumerate() {
            if conflicting.contains(&index) {
                continue;
            }
            plan.moved_files.extend(item.moved_files);
            plan.rules.push(item.rule);
        }
        plan
    }

    fn remap_module(&self, db: &dyn Db, module: Module<'_>) -> ModuleRemap {
        let name = module.name(db);
        let mut best: Option<(usize, ModuleName)> = None;
        let mut ambiguous = false;

        for rule in &self.rules {
            let Some(new_name) = rule.rewrite_name(name) else {
                continue;
            };
            match rule.applies_to_module(db, module) {
                Some(true) => {
                    let specificity = rule.old_name.components().count();
                    if best
                        .as_ref()
                        .is_none_or(|(best_specificity, _)| specificity > *best_specificity)
                    {
                        best = Some((specificity, new_name));
                    }
                }
                Some(false) => {}
                None => ambiguous = true,
            }
        }

        if let Some((_, new_name)) = best {
            ModuleRemap::Renamed(new_name)
        } else if ambiguous {
            ModuleRemap::Ambiguous
        } else {
            ModuleRemap::Unchanged
        }
    }

    fn source_mentions_any(&self, source: &str) -> bool {
        self.rules.iter().any(|rule| {
            contains_identifier(source, rule.old_name.first_component())
                || contains_identifier(source, rule.old_name.last_component())
        })
    }
}

struct ResolvedRename {
    rule: RenameRule,
    moved_files: FxHashSet<File>,
}

impl ResolvedRename {
    fn from_path_rename(db: &dyn Db, rename: &PathRename, files: &FxHashSet<File>) -> Option<Self> {
        let old_path = SystemPath::absolute(&rename.old_path, db.system().current_directory());
        let new_path = SystemPath::absolute(&rename.new_path, db.system().current_directory());

        match rename.kind {
            PathRenameKind::File => {
                let extension = old_path.extension()?;
                if !matches!(extension, "py" | "pyi")
                    || new_path.extension() != Some(extension)
                    || old_path.file_stem() == Some("__init__")
                    || new_path.file_stem() == Some("__init__")
                {
                    return None;
                }
                let file = system_path_to_file(db, &old_path).ok()?;
                let old_name = file_to_module(db, file)?.name(db).clone();
                let new_name = module_name_for_path(db, &new_path, PathRenameKind::File)?;
                if old_name == new_name {
                    return None;
                }
                Some(Self {
                    rule: RenameRule {
                        old_name,
                        new_name,
                        source: RenameSource::Files {
                            files: FxHashSet::from_iter([file]),
                        },
                    },
                    moved_files: FxHashSet::from_iter([file]),
                })
            }
            PathRenameKind::Directory => {
                if !db.system().is_directory(&old_path) || new_path.starts_with(&old_path) {
                    return None;
                }
                let old_name = module_name_for_path(db, &old_path, PathRenameKind::Directory)?;
                let new_name = module_name_for_path(db, &new_path, PathRenameKind::Directory)?;
                if old_name == new_name {
                    return None;
                }
                let moved_files: FxHashSet<_> = files
                    .iter()
                    .copied()
                    .filter(|file| {
                        file.path(db)
                            .as_system_path()
                            .is_some_and(|path| path.starts_with(&old_path))
                    })
                    .collect();
                if moved_files.is_empty() {
                    return None;
                }
                Some(Self {
                    rule: RenameRule {
                        old_name,
                        new_name,
                        source: RenameSource::Directory { old_root: old_path },
                    },
                    moved_files,
                })
            }
        }
    }
}

struct RenameRule {
    old_name: ModuleName,
    new_name: ModuleName,
    source: RenameSource,
}

impl RenameRule {
    fn rewrite_name(&self, name: &ModuleName) -> Option<ModuleName> {
        if name == &self.old_name {
            return Some(self.new_name.clone());
        }
        if !matches!(self.source, RenameSource::Directory { .. }) {
            return None;
        }
        let suffix = name.relative_to(&self.old_name)?;
        let mut new_name = self.new_name.clone();
        new_name.extend(&suffix);
        Some(new_name)
    }

    /// Returns `None` when the matching module is an aggregate namespace without a concrete file.
    fn applies_to_module(&self, db: &dyn Db, module: Module<'_>) -> Option<bool> {
        let file = resolve_real_module_confident(db, module.name(db))
            .and_then(|module| module.file(db))
            .or_else(|| module.file(db));
        match &self.source {
            RenameSource::Files { files } => Some(file.is_some_and(|file| files.contains(&file))),
            RenameSource::Directory { old_root } => {
                let file = file?;
                Some(
                    file.path(db)
                        .as_system_path()
                        .is_some_and(|path| path.starts_with(old_root)),
                )
            }
        }
    }

    fn conflicts_with(&self, db: &dyn Db, other: &Self) -> bool {
        if self.old_name == other.old_name {
            return self.new_name != other.new_name;
        }
        if self.new_name == other.new_name {
            return true;
        }

        self.contains_source_of(db, other)
            .is_some_and(|expected| expected != other.new_name)
            || other
                .contains_source_of(db, self)
                .is_some_and(|expected| expected != self.new_name)
    }

    fn contains_source_of(&self, db: &dyn Db, other: &Self) -> Option<ModuleName> {
        let RenameSource::Directory { old_root } = &self.source else {
            return None;
        };
        let contained = match &other.source {
            RenameSource::Files { files } => files.iter().any(|file| {
                file.path(db)
                    .as_system_path()
                    .is_some_and(|path| path.starts_with(old_root))
            }),
            RenameSource::Directory {
                old_root: other_root,
            } => other_root.starts_with(old_root),
        };
        contained
            .then(|| self.rewrite_name(&other.old_name))
            .flatten()
    }
}

enum RenameSource {
    Files { files: FxHashSet<File> },
    Directory { old_root: SystemPathBuf },
}

enum ModuleRemap {
    Unchanged,
    Renamed(ModuleName),
    Ambiguous,
}

fn module_name_for_path(
    db: &dyn Db,
    path: &SystemPath,
    kind: PathRenameKind,
) -> Option<ModuleName> {
    let path = SystemPath::absolute(path, db.system().current_directory());
    for search_path in search_paths(db, ModuleResolveMode::StubsAllowed) {
        let Some(root) = search_path.as_system_path() else {
            continue;
        };
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        if search_path.is_standard_library() {
            return None;
        }
        let (directory, file_name) = match kind {
            PathRenameKind::File => (relative.parent()?, Some(path.file_stem()?)),
            PathRenameKind::Directory => (relative, None),
        };
        let mut components: Vec<_> = directory
            .components()
            .map(|component| component.as_str())
            .collect();
        if let Some(first) = components.first_mut() {
            *first = first.strip_suffix("-stubs").unwrap_or(first);
        }
        components.extend(file_name);
        let Some(name) = ModuleName::from_components(components) else {
            continue;
        };

        return Some(name);
    }
    None
}

fn rename_edits_for_file(db: &dyn Db, file: File, plan: &RenamePlan) -> Option<FileRenameResult> {
    let source = source_text(db, file);
    if source.read_error().is_some() {
        return None;
    }
    if !plan.moved_files.contains(&file) && !plan.source_mentions_any(source.as_str()) {
        return Some(FileRenameResult::default());
    }

    let parsed = ruff_db::parsed::parsed_module(db, file);
    let module = parsed.load(db);
    let importing_module = file_to_module(db, file).map(|importing| ImportingModule {
        name: importing.name(db).clone(),
        is_package: importing.kind(db).is_package(),
        moved: plan.moved_files.contains(&file),
    });
    let model = SemanticModel::new(db, file);
    let mut visitor = ModuleRenameVisitor {
        db,
        model: &model,
        tokens: module.tokens(),
        source: source.as_str(),
        plan,
        importing_module: importing_module.as_ref(),
        edits: Vec::new(),
        required_edits_from: FxHashSet::default(),
        changed_implicit_bindings: FxHashSet::default(),
        import_binding_stability: FxHashMap::default(),
        collecting_imports: true,
        supported: true,
    };
    // Collect every changed binding first so uses that precede their import are validated too.
    AnyNodeRef::from(module.syntax()).visit_source_order(&mut visitor);
    visitor.collecting_imports = false;
    AnyNodeRef::from(module.syntax()).visit_source_order(&mut visitor);
    visitor.finish(file)
}

#[derive(Default)]
struct FileRenameResult {
    edits: Vec<FileRenameEdit>,
    required_edits_from: FxHashSet<File>,
}

struct ImportingModule {
    name: ModuleName,
    is_package: bool,
    moved: bool,
}

struct ModuleRenameVisitor<'a, 'db> {
    db: &'db dyn Db,
    model: &'a SemanticModel<'db>,
    tokens: &'a Tokens,
    source: &'a str,
    plan: &'a RenamePlan,
    importing_module: Option<&'a ImportingModule>,
    edits: Vec<(TextRange, String)>,
    required_edits_from: FxHashSet<File>,
    changed_implicit_bindings: FxHashSet<String>,
    import_binding_stability: FxHashMap<(FileScopeId, ScopedPlaceId), bool>,
    collecting_imports: bool,
    supported: bool,
}

impl<'a> SourceOrderVisitor<'a> for ModuleRenameVisitor<'_, '_> {
    fn enter_node(&mut self, node: AnyNodeRef<'a>) -> TraversalSignal {
        if !self.supported {
            return TraversalSignal::Skip;
        }
        if self.collecting_imports {
            return match node {
                AnyNodeRef::StmtImport(import) => {
                    self.handle_import(import);
                    TraversalSignal::Skip
                }
                AnyNodeRef::StmtImportFrom(import) => {
                    self.handle_import_from(import);
                    TraversalSignal::Skip
                }
                _ => TraversalSignal::Traverse,
            };
        }
        match node {
            AnyNodeRef::StmtImport(_) | AnyNodeRef::StmtImportFrom(_) => TraversalSignal::Skip,
            AnyNodeRef::StmtGlobal(global) => {
                self.reject_changed_declarations(&global.names);
                TraversalSignal::Skip
            }
            AnyNodeRef::StmtNonlocal(nonlocal) => {
                self.reject_changed_declarations(&nonlocal.names);
                TraversalSignal::Skip
            }
            AnyNodeRef::ExprName(name) => {
                if self.handle_name(name) {
                    TraversalSignal::Skip
                } else {
                    TraversalSignal::Traverse
                }
            }
            AnyNodeRef::ExprAttribute(attribute) => {
                if self.handle_attribute(attribute) {
                    TraversalSignal::Skip
                } else {
                    TraversalSignal::Traverse
                }
            }
            _ => TraversalSignal::Traverse,
        }
    }
}

impl ModuleRenameVisitor<'_, '_> {
    fn reject_changed_declarations(&mut self, names: &[ast::Identifier]) {
        if names
            .iter()
            .any(|name| self.changed_implicit_bindings.contains(name.as_str()))
        {
            self.supported = false;
        }
    }

    fn record_import_binding(&mut self, alias: &ast::Alias, is_stable: bool) {
        let Some(definition) = semantic_index(self.db, self.model.file()).try_definition(alias)
        else {
            self.supported = false;
            return;
        };
        let binding = (definition.file_scope(self.db), definition.place(self.db));
        if self
            .import_binding_stability
            .insert(binding, is_stable)
            .is_some_and(|previously_stable| previously_stable != is_stable)
        {
            self.supported = false;
        }
    }

    fn finish(mut self, file: File) -> Option<FileRenameResult> {
        if !self.supported {
            return None;
        }
        self.edits.sort_by(|left, right| {
            left.0
                .start()
                .cmp(&right.0.start())
                .then_with(|| left.0.end().cmp(&right.0.end()))
                .then_with(|| left.1.cmp(&right.1))
        });
        self.edits.dedup();
        if self.edits.windows(2).any(|edits| {
            edits[0].0.start() == edits[1].0.start() || edits[0].0.end() > edits[1].0.start()
        }) {
            return None;
        }
        Some(FileRenameResult {
            edits: self
                .edits
                .into_iter()
                .map(|(range, new_text)| FileRenameEdit {
                    file,
                    range,
                    new_text,
                })
                .collect(),
            required_edits_from: self.required_edits_from,
        })
    }

    fn handle_import(&mut self, import: &ast::StmtImport) {
        let mut edits = Vec::new();
        for alias in &import.names {
            if alias.asname.is_some() {
                self.record_import_binding(alias, true);
            }
            let Some(module) = self.model.resolve_module(Some(alias.name.as_str()), 0) else {
                continue;
            };
            match self.plan.remap_module(self.db, module) {
                ModuleRemap::Unchanged => {}
                ModuleRemap::Renamed(new_name) => {
                    // An unaliased dotted import binds its first component. Changing that
                    // component can invalidate independent uses of the implicit parent binding.
                    if alias.asname.is_none()
                        && alias.name.as_str().contains('.')
                        && module.name(self.model.db()).first_component()
                            != new_name.first_component()
                    {
                        self.supported = false;
                        return;
                    }
                    if alias.asname.is_none()
                        && module.name(self.model.db()).first_component()
                            != new_name.first_component()
                    {
                        self.changed_implicit_bindings
                            .insert(module.name(self.model.db()).first_component().to_string());
                        self.record_import_binding(alias, false);
                    }
                    edits.push((alias.name.range, new_name.as_str().to_string()));
                }
                ModuleRemap::Ambiguous => {
                    self.supported = false;
                    return;
                }
            }
        }
        for (range, new_text) in edits {
            self.push_if_changed(range, new_text);
        }
    }

    fn handle_import_from(&mut self, import: &ast::StmtImportFrom) {
        if import.level > 0
            && self
                .importing_module
                .is_some_and(|importing| importing.moved)
        {
            self.supported = false;
            return;
        }
        let Ok(old_parent) =
            ModuleName::from_import_statement(self.model.db(), self.model.file(), import)
        else {
            return;
        };
        let parent_module = self.model.resolve_module(
            import.module.as_ref().map(ast::Identifier::as_str),
            import.level,
        );
        let parent_remap = parent_module.map_or(ModuleRemap::Unchanged, |module| {
            self.plan.remap_module(self.db, module)
        });
        let default_parent = match &parent_remap {
            ModuleRemap::Unchanged => Some(old_parent.clone()),
            ModuleRemap::Renamed(new_parent) => Some(new_parent.clone()),
            ModuleRemap::Ambiguous => None,
        };

        let mut statement_parent = None;
        let mut alias_edits = Vec::new();
        for alias in &import.names {
            if alias.asname.is_some() {
                self.record_import_binding(alias, true);
            }
            // A module whose canonical parent differs from the statement parent is a re-export,
            // even when the imported member happens to have the same name.
            let alias_module = module_from_type(self.model, alias).filter(|module| {
                alias.name.as_str() == module.name(self.model.db()).last_component()
            });
            let alias_parent = if let Some(module) = alias_module {
                let is_direct_child =
                    module.name(self.model.db()).parent().as_ref() == Some(&old_parent);
                let preserves_public_name = !is_direct_child
                    && imported_symbol_has_explicit_alias(self.model, import, alias.name.as_str());
                match self.plan.remap_module(self.db, module) {
                    ModuleRemap::Renamed(new_name) => {
                        if alias.name.as_str() != new_name.last_component() {
                            let stable = preserves_public_name || alias.asname.is_some();
                            if !stable {
                                self.changed_implicit_bindings
                                    .insert(alias.name.as_str().to_string());
                            }
                            if alias.asname.is_none() {
                                self.record_import_binding(alias, stable);
                            }
                            if !preserves_public_name {
                                if !is_direct_child {
                                    let Some(file) =
                                        parent_module.and_then(|module| module.file(self.db))
                                    else {
                                        self.supported = false;
                                        return;
                                    };
                                    self.required_edits_from.insert(file);
                                }
                                alias_edits.push((
                                    alias.name.range,
                                    new_name.last_component().to_string(),
                                ));
                            }
                        }
                        if is_direct_child {
                            let Some(new_parent) = new_name.parent() else {
                                self.supported = false;
                                return;
                            };
                            new_parent
                        } else {
                            default_parent.clone().unwrap_or_else(|| old_parent.clone())
                        }
                    }
                    ModuleRemap::Unchanged => {
                        default_parent.clone().unwrap_or_else(|| old_parent.clone())
                    }
                    ModuleRemap::Ambiguous => {
                        self.supported = false;
                        return;
                    }
                }
            } else if let Some(default_parent) = &default_parent {
                default_parent.clone()
            } else {
                self.supported = false;
                return;
            };

            if statement_parent
                .as_ref()
                .is_some_and(|parent| parent != &alias_parent)
            {
                self.supported = false;
                return;
            }
            statement_parent.get_or_insert(alias_parent);
        }

        let statement_parent = statement_parent.unwrap_or(old_parent.clone());
        let mut edits = Vec::new();
        if statement_parent != old_parent {
            let Some(range) = import_from_module_range(self.tokens, import) else {
                self.supported = false;
                return;
            };
            let Some(new_text) =
                render_import_from_module(import, &statement_parent, self.importing_module)
            else {
                self.supported = false;
                return;
            };
            edits.push((range, new_text));
        }
        edits.extend(alias_edits);
        for (range, new_text) in edits {
            self.push_if_changed(range, new_text);
        }
    }

    fn handle_name(&mut self, name: &ast::ExprName) -> bool {
        if self.changed_implicit_bindings.contains(name.id.as_str())
            && explicit_import_alias_for_name(self.model, name).is_none()
        {
            self.supported = false;
            return true;
        }
        let Some(module) = module_from_type(self.model, name) else {
            return false;
        };
        let old_name = module.name(self.model.db());
        match self.plan.remap_module(self.db, module) {
            ModuleRemap::Unchanged => false,
            ModuleRemap::Ambiguous if module.file(self.db).is_none() => true,
            ModuleRemap::Ambiguous => {
                self.supported = false;
                true
            }
            ModuleRemap::Renamed(new_name) => {
                let Some(has_explicit_alias) = explicit_import_alias_for_name(self.model, name)
                else {
                    self.supported = false;
                    return true;
                };
                if has_explicit_alias {
                    return true;
                }
                let new_text = if name.id.as_str() == old_name.as_str() {
                    Some(new_name.as_str())
                } else if name.id.as_str() == old_name.last_component() {
                    Some(new_name.last_component())
                } else {
                    None
                };
                if let Some(new_text) = new_text {
                    if !name.ctx.is_load() && new_text.contains('.') {
                        self.supported = false;
                        return true;
                    }
                    self.push_if_changed(name.range, new_text.to_string());
                }
                true
            }
        }
    }

    fn handle_attribute(&mut self, attribute: &ast::ExprAttribute) -> bool {
        let Some(module) = module_from_type(self.model, attribute) else {
            return false;
        };
        let old_name = module.name(self.model.db());
        match self.plan.remap_module(self.db, module) {
            // The complete module expression is unchanged. Do not descend into an aggregate
            // namespace root, which cannot be attributed to one physical portion on its own.
            ModuleRemap::Unchanged => module.file(self.db).is_none(),
            ModuleRemap::Ambiguous if module.file(self.db).is_none() => true,
            ModuleRemap::Ambiguous => {
                self.supported = false;
                true
            }
            ModuleRemap::Renamed(new_name) => {
                if let Some(new_text) = rewritten_attribute_path(
                    self.model, attribute, old_name, &new_name, self.plan, self.db,
                ) {
                    if !self.push_attribute_rewrite(attribute, &new_text) {
                        self.supported = false;
                    }
                } else if !attribute_has_explicit_alias(self.model, attribute) {
                    self.supported = false;
                }
                true
            }
        }
    }

    fn push_if_changed(&mut self, range: TextRange, new_text: String) {
        let start = usize::from(range.start());
        let end = usize::from(range.end());
        if self.source.get(start..end) != Some(new_text.as_str()) {
            self.edits.push((range, new_text));
        }
    }

    fn push_attribute_rewrite(&mut self, attribute: &ast::ExprAttribute, new_text: &str) -> bool {
        let Some(ranges) = attribute_path_ranges(attribute) else {
            return false;
        };
        let components: Vec<_> = new_text.split('.').collect();
        if ranges.len() != components.len() {
            return false;
        }
        // Component edits preserve any comments and parentheses between identifiers.
        for (range, component) in ranges.into_iter().zip(components) {
            self.push_if_changed(range, component.to_string());
        }
        true
    }
}

fn attribute_has_explicit_alias(model: &SemanticModel<'_>, attribute: &ast::ExprAttribute) -> bool {
    definitions_for_attribute_with_alias_resolution(
        model,
        attribute,
        ImportAliasResolution::PreserveAliases,
    )
    .iter()
    .any(|resolved| is_explicit_import_alias(model, resolved))
}

fn module_from_type<'db, T: HasType>(
    model: &SemanticModel<'db>,
    expression: &T,
) -> Option<Module<'db>> {
    let Type::ModuleLiteral(literal) = expression.inferred_type(model)? else {
        return None;
    };
    Some(literal.module(model.db()))
}

fn explicit_import_alias_for_name(model: &SemanticModel<'_>, name: &ast::ExprName) -> Option<bool> {
    let definitions = definitions_for_name(
        model,
        name.id.as_str(),
        name.into(),
        ImportAliasResolution::PreserveAliases,
    );
    let all_explicit_aliases = definitions
        .iter()
        .all(|resolved| is_explicit_import_alias(model, resolved));
    match definitions.len() {
        0 => None,
        1 => Some(all_explicit_aliases),
        _ => all_explicit_aliases.then_some(true),
    }
}

fn imported_symbol_has_explicit_alias(
    model: &SemanticModel<'_>,
    import: &ast::StmtImportFrom,
    name: &str,
) -> bool {
    definitions_for_imported_symbol(model, import, name, ImportAliasResolution::PreserveAliases)
        .iter()
        .any(|resolved| is_explicit_import_alias(model, resolved))
}

fn is_explicit_import_alias(model: &SemanticModel<'_>, resolved: &ResolvedDefinition<'_>) -> bool {
    let Some(definition) = resolved.definition() else {
        return false;
    };
    let db = model.db();
    let module = ruff_db::parsed::parsed_module(db, definition.file(db)).load(db);
    match definition.kind(db) {
        DefinitionKind::Import(import) => import.alias(&module).asname.is_some(),
        DefinitionKind::ImportFrom(import) => import.alias(&module).asname.is_some(),
        _ => false,
    }
}

fn rewritten_attribute_path(
    model: &SemanticModel<'_>,
    attribute: &ast::ExprAttribute,
    old_name: &ModuleName,
    new_name: &ModuleName,
    plan: &RenamePlan,
    db: &dyn Db,
) -> Option<String> {
    let root = root_name_of(&attribute.value)?;
    let root_module = module_from_type(model, root)?;
    let root_old = root_module.name(model.db());
    let has_explicit_alias = explicit_import_alias_for_name(model, root)?;
    let removed_components = old_name.relative_to(root_old)?.components().count();
    let mut root_new = new_name.clone();
    for _ in 0..removed_components {
        root_new = root_new.parent()?;
    }
    if &root_new != root_old
        && !matches!(
            plan.remap_module(db, root_module),
            ModuleRemap::Renamed(remapped) if remapped == root_new
        )
    {
        return None;
    }
    if !has_explicit_alias && root.id.as_str() == root_old.as_str() {
        return Some(new_name.as_str().to_string());
    }

    let suffix = new_name.relative_to(&root_new)?;
    let root_text = if has_explicit_alias {
        root.id.as_str()
    } else if root.id.as_str() == root_old.last_component() {
        root_new.last_component()
    } else {
        if &root_new != root_old {
            return None;
        }
        root.id.as_str()
    };
    if suffix.as_str().is_empty() {
        Some(root_text.to_string())
    } else {
        Some(format!("{root_text}.{suffix}"))
    }
}

fn attribute_path_ranges(attribute: &ast::ExprAttribute) -> Option<Vec<TextRange>> {
    let mut ranges = vec![attribute.attr.range];
    let mut expression = &*attribute.value;
    loop {
        match expression {
            ast::Expr::Name(name) => {
                ranges.push(name.range);
                ranges.reverse();
                return Some(ranges);
            }
            ast::Expr::Attribute(attribute) => {
                ranges.push(attribute.attr.range);
                expression = &attribute.value;
            }
            _ => return None,
        }
    }
}

fn root_name_of(expression: &ast::Expr) -> Option<&ast::ExprName> {
    match expression {
        ast::Expr::Name(name) => Some(name),
        ast::Expr::Attribute(attribute) => root_name_of(&attribute.value),
        _ => None,
    }
}

fn import_from_module_range(tokens: &Tokens, import: &ast::StmtImportFrom) -> Option<TextRange> {
    let mut after_from = false;
    let mut first = None;
    let mut last = None;

    for token in tokens.in_range(import.range) {
        match token.kind() {
            TokenKind::From => after_from = true,
            TokenKind::Import if after_from => break,
            TokenKind::Dot | TokenKind::Ellipsis | TokenKind::Name if after_from => {
                first.get_or_insert(token.start());
                last = Some(token.end());
            }
            _ => {}
        }
    }
    Some(TextRange::new(first?, last?))
}

fn render_import_from_module(
    import: &ast::StmtImportFrom,
    new_name: &ModuleName,
    importing_module: Option<&ImportingModule>,
) -> Option<String> {
    if import.level == 0 {
        return Some(new_name.as_str().to_string());
    }

    let importing_module = importing_module?;
    let anchor = if importing_module.is_package {
        Some(importing_module.name.clone())
    } else {
        importing_module.name.parent()
    };
    let Some(anchor) = anchor else {
        return Some(new_name.as_str().to_string());
    };

    for (depth, base) in anchor.ancestors().enumerate() {
        let mut rendered = ".".repeat(depth + 1);
        if new_name == &base {
            return Some(rendered);
        }
        if let Some(relative) = new_name.relative_to(&base) {
            rendered.push_str(relative.as_str());
            return Some(rendered);
        }
    }
    Some(new_name.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruff_db::Db as _;
    use ruff_db::files::system_path_to_file;
    use ruff_db::source::source_text;
    use ruff_db::system::{DbWithWritableSystem, SystemPathBuf};
    use ruff_python_ast::PythonVersion;
    use ty_module_resolver::SearchPathSettings;
    use ty_project::{ProjectMetadata, TestDb};
    use ty_python_core::platform::PythonPlatform;
    use ty_python_core::program::{FallibleStrategy, Program, ProgramSettings};
    use ty_python_semantic::PythonVersionWithSource;

    #[test]
    fn semantic_file_move_preserves_aliases_reexports_and_shadowed_names() {
        let db = create_test_db(&[
            ("/old_pkg/__init__.py", ""),
            ("/old_pkg/old.py", "import helper as helper\n"),
            ("/new_pkg/__init__.py", ""),
            ("/helper.py", "x = 1\n"),
            ("/old.py", "x = 1\n"),
            ("/facade.py", "import old_pkg.old as stable\n"),
            (
                "/consumer.py",
                "def qualified():\n    import old_pkg.old\n    return (old_pkg  # preserved\n            ).old.helper.x\nimport old_pkg.old as direct_stable\nfrom facade import stable\ndef local():\n    from old_pkg import old\n    return old.helper.x, direct_stable.helper.x, stable.helper.x\ndef explicit_alias():\n    import old_pkg.old as old\n    return old.helper.x\ndef explicit_from_alias():\n    from old_pkg import old as old\n    return old.helper.x\ndef top_level_alias():\n    import old as old\n    return old.x\ndef shadowed(old_pkg):\n    return old_pkg.old\n",
            ),
        ]);

        let edits = will_rename(
            &db,
            &[
                PathRename::file("/old_pkg/old.py".into(), "/old_pkg/new.py".into()),
                PathRename::file("/old.py".into(), "/new_pkg/nested.py".into()),
            ],
        );
        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let facade = system_path_to_file(&db, "/facade.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, consumer),
            "def qualified():\n    import old_pkg.new\n    return (old_pkg  # preserved\n            ).new.helper.x\nimport old_pkg.new as direct_stable\nfrom facade import stable\ndef local():\n    from old_pkg import new\n    return new.helper.x, direct_stable.helper.x, stable.helper.x\ndef explicit_alias():\n    import old_pkg.new as old\n    return old.helper.x\ndef explicit_from_alias():\n    from old_pkg import new as old\n    return old.helper.x\ndef top_level_alias():\n    import new_pkg.nested as old\n    return old.x\ndef shadowed(old_pkg):\n    return old_pkg.old\n"
        );
        assert_eq!(
            apply_edits(&db, &edits, facade),
            "import old_pkg.new as stable\n"
        );
    }

    #[test]
    fn directory_move_supports_regular_and_namespace_packages() {
        for (initializer, old, new) in [
            (Some(("/old_pkg/__init__.py", "")), "/old_pkg", "/new_pkg"),
            (None, "/old_ns", "/new_ns"),
        ] {
            let is_regular = initializer.is_some();
            let mut files = vec![
                ("/consumer.py", "import old_pkg.sub as sub\nprint(sub.x)\n"),
                ("/old_pkg/sub.py", "x = 1\n"),
            ];
            if let Some(initializer) = initializer {
                files.push(initializer);
            } else {
                files = vec![
                    ("/consumer.py", "import old_ns.sub as sub\nprint(sub.x)\n"),
                    ("/old_ns/sub.py", "x = 1\n"),
                ];
            }
            let db = create_test_db(&files);
            let edits = will_rename(&db, &[PathRename::directory(old.into(), new.into())]);
            let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
            let expected = if is_regular {
                "import new_pkg.sub as sub\nprint(sub.x)\n"
            } else {
                "import new_ns.sub as sub\nprint(sub.x)\n"
            };
            assert_eq!(apply_edits(&db, &edits, consumer), expected);
        }

        let db = create_test_db(&[
            ("/old_pkg/__init__.py", "from .sub import thing\n"),
            ("/old_pkg/sub/__init__.py", ""),
            ("/old_pkg/sub/thing.py", "x = 1\n"),
            (
                "/old_pkg/relative.py",
                "from .sub import thing\nimport old_pkg\n",
            ),
            (
                "/consumer.py",
                "import old_pkg as old_pkg\nfrom old_pkg import thing\nprint(old_pkg.sub.thing.x, thing.x)\n",
            ),
        ]);
        let edits = will_rename(
            &db,
            &[PathRename::directory("/old_pkg".into(), "/new_pkg".into())],
        );
        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let relative = system_path_to_file(&db, "/old_pkg/relative.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, consumer),
            "import new_pkg as old_pkg\nfrom new_pkg import thing\nprint(old_pkg.sub.thing.x, thing.x)\n"
        );
        assert_eq!(
            apply_edits(&db, &edits, relative),
            "from .sub import thing\nimport old_pkg\n"
        );

        let mut db = create_test_db(&[
            ("/stubs/pkg/__init__.pyi", "x: int\n"),
            ("/src/pkg/__init__.py", "x = 1\n"),
            ("/consumer/consumer.py", "import pkg\nprint(pkg.x)\n"),
        ]);
        configure_search_paths(
            &mut db,
            vec!["/stubs".into(), "/src".into(), "/consumer".into()],
        );
        let edits = will_rename(
            &db,
            &[PathRename::directory(
                "/src/pkg".into(),
                "/src/new_pkg".into(),
            )],
        );
        let consumer = system_path_to_file(&db, "/consumer/consumer.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, consumer),
            "import new_pkg\nprint(new_pkg.x)\n"
        );

        let db = create_test_db(&[
            ("/a/__init__.py", ""),
            ("/a/old_pkg/__init__.py", ""),
            ("/a/old_pkg/mod.py", "x = 1\n"),
            ("/b/__init__.py", ""),
            ("/consumer.py", "from a.old_pkg import mod\nprint(mod.x)\n"),
        ]);
        let edits = will_rename(
            &db,
            &[PathRename::directory(
                "/a/old_pkg".into(),
                "/b/new_pkg".into(),
            )],
        );
        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, consumer),
            "from b.new_pkg import mod\nprint(mod.x)\n"
        );
    }

    #[test]
    fn batch_handles_source_stub_provenance_and_isolates_conflicts() {
        let mut db = create_test_db(&[
            ("/old_a.py", "x = 1\n"),
            ("/old_b.py", "x = 1\n"),
            ("/old_b.pyi", "x: int\n"),
            ("/partial.py", "x = 1\n"),
            ("/partial.pyi", "x: int\n"),
            ("/source_only.py", "x = 1\n"),
            ("/source_only.pyi", "x: int\n"),
            ("/foo.py", "file_value = 1\n"),
            ("/foo/__init__.py", "package_value = 1\n"),
            ("/stubs/pep-stubs/__init__.pyi", ""),
            ("/stubs/pep-stubs/old.pyi", "x: int\n"),
            ("/site-packages/old.py", "x = 1\n"),
            (
                "/consumer.py",
                "import old_a\nimport old_b\nimport foo\nimport old\nimport partial\nimport pep.old\nimport source_only\nprint(old_a.x, old_b.x, foo.package_value, old.x, partial.x, pep.old.x, source_only.x)\n",
            ),
        ]);
        configure_search_paths(
            &mut db,
            vec!["/stubs".into(), "/".into(), "/site-packages".into()],
        );
        let edits = will_rename(
            &db,
            &[
                PathRename::file("/old_a.py".into(), "/new_a.py".into()),
                PathRename::file("/old_a.py".into(), "/other_a.py".into()),
                PathRename::file("/old_b.py".into(), "/new_b.py".into()),
                PathRename::file("/old_b.pyi".into(), "/new_b.pyi".into()),
                PathRename::file("/partial.pyi".into(), "/renamed_partial.pyi".into()),
                PathRename::file(
                    "/stubs/pep-stubs/old.pyi".into(),
                    "/stubs/pep-stubs/new.pyi".into(),
                ),
                PathRename::file("/source_only.py".into(), "/renamed_source.py".into()),
                PathRename::file("/foo.py".into(), "/baz.py".into()),
                PathRename::directory("/foo".into(), "/bar".into()),
                PathRename::file(
                    "/site-packages/old.py".into(),
                    "/site-packages/new.py".into(),
                ),
            ],
        );
        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, consumer),
            "import old_a\nimport new_b\nimport bar\nimport new\nimport partial\nimport pep.new\nimport renamed_source\nprint(old_a.x, new_b.x, bar.package_value, new.x, partial.x, pep.new.x, renamed_source.x)\n"
        );
    }

    #[test]
    fn path_mapping_uses_search_precedence_and_ignores_noop_items() {
        let mut db = create_test_db(&[
            ("/src/pkg/__init__.py", ""),
            ("/src/pkg/old.py", "x = 1\n"),
            ("/one/noop/sibling.py", "x = 1\n"),
            (
                "/one/noop/mod.py",
                "from . import sibling\nimport pkg.old\nprint(pkg.old.x, sibling.x)\n",
            ),
            ("/two/placeholder.py", ""),
        ]);
        configure_search_paths(&mut db, vec!["/src".into(), "/one".into(), "/two".into()]);

        let edits = will_rename(
            &db,
            &[
                PathRename::file("/src/pkg/old.py".into(), "/src/pkg/new.py".into()),
                PathRename::file("/one/noop/mod.py".into(), "/two/noop/mod.py".into()),
            ],
        );
        let consumer = system_path_to_file(&db, "/one/noop/mod.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, consumer),
            "from . import sibling\nimport pkg.new\nprint(pkg.new.x, sibling.x)\n"
        );
    }

    #[test]
    fn unsupported_rewrite_discards_only_its_file() {
        let db = create_test_db(&[
            ("/old_pkg/__init__.py", "other = 1\n"),
            ("/old_pkg/moved.py", "x = 1\n"),
            ("/new_pkg/__init__.py", ""),
            ("/a/__init__.py", ""),
            ("/a/b/__init__.py", "from . import c\n"),
            ("/a/b/c.py", "x = 1\n"),
            ("/old.py", "x = 1\n"),
            ("/fallback.py", "old = None\n"),
            (
                "/safe.py",
                "import old_pkg.moved as moved\nprint(moved.x)\n",
            ),
            (
                "/unsupported.py",
                "from old_pkg import moved, other\nprint(moved.x, other)\n",
            ),
            (
                "/unsupported_reexport_consumer.py",
                "from unsupported import moved\nprint(moved.x)\n",
            ),
            (
                "/unchanged_alias.py",
                "import old_pkg.moved as loaded\nimport old_pkg as stable\nfrom old_pkg import moved\nprint(stable.moved.x, moved.x, loaded.x)\n",
            ),
            (
                "/dotted_parent.py",
                "import old_pkg.moved\nprint(old_pkg)\n",
            ),
            ("/old_pkg/facade.py", "from old_pkg import moved\n"),
            (
                "/explicit_facade.py",
                "from old_pkg import moved as moved\n",
            ),
            (
                "/explicit_facade_consumer.py",
                "from explicit_facade import moved\nprint(moved.x)\n",
            ),
            (
                "/qualified_explicit_reexport.py",
                "import old_pkg.moved as direct\nimport explicit_facade\nprint(direct.x, explicit_facade.moved.x)\n",
            ),
            (
                "/qualified_reexport.py",
                "import old_pkg.facade\nprint(old_pkg.facade.moved.x)\n",
            ),
            (
                "/same_name_reexport.py",
                "from old_pkg.facade import moved\nprint(moved.x)\n",
            ),
            (
                "/store.py",
                "import old as loaded\nprint((old := loaded).x)\n",
            ),
            (
                "/mixed_import.py",
                "if flag:\n    import old\nelse:\n    import fallback as old\n",
            ),
            (
                "/mixed_from_import.py",
                "if flag:\n    import old\nelse:\n    from fallback import old as old\n",
            ),
            ("/bound_parent.py", "from a import b\nprint(b.c.x)\n"),
            (
                "/rebound.py",
                "def use():\n    return old\nif flag:\n    old = fallback\nelse:\n    import old\n",
            ),
            (
                "/global.py",
                "import old\ndef clear():\n    global old\n    del old\n",
            ),
        ]);
        let edits = will_rename(
            &db,
            &[
                PathRename::file("/old_pkg/moved.py".into(), "/new_pkg/new.py".into()),
                PathRename::file("/old.py".into(), "/new_pkg/top.py".into()),
                PathRename::file("/a/b/c.py".into(), "/a/d.py".into()),
            ],
        );
        let safe = system_path_to_file(&db, "/safe.py").unwrap();
        let facade = system_path_to_file(&db, "/old_pkg/facade.py").unwrap();
        let same_name_reexport = system_path_to_file(&db, "/same_name_reexport.py").unwrap();
        let explicit_facade = system_path_to_file(&db, "/explicit_facade.py").unwrap();
        let explicit_facade_consumer =
            system_path_to_file(&db, "/explicit_facade_consumer.py").unwrap();
        let qualified_explicit_reexport =
            system_path_to_file(&db, "/qualified_explicit_reexport.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, safe),
            "import new_pkg.new as moved\nprint(moved.x)\n"
        );
        for path in [
            "/unsupported.py",
            "/unsupported_reexport_consumer.py",
            "/unchanged_alias.py",
            "/dotted_parent.py",
            "/qualified_reexport.py",
            "/store.py",
            "/mixed_import.py",
            "/mixed_from_import.py",
            "/rebound.py",
            "/global.py",
            "/bound_parent.py",
        ] {
            let file = system_path_to_file(&db, path).unwrap();
            assert_eq!(
                apply_edits(&db, &edits, file),
                source_text(&db, file).as_str()
            );
        }
        assert_eq!(
            apply_edits(&db, &edits, same_name_reexport),
            "from old_pkg.facade import new\nprint(new.x)\n"
        );
        assert_eq!(
            apply_edits(&db, &edits, facade),
            "from new_pkg import new\n"
        );
        assert_eq!(
            apply_edits(&db, &edits, explicit_facade),
            "from new_pkg import new as moved\n"
        );
        assert_eq!(
            apply_edits(&db, &edits, explicit_facade_consumer),
            "from explicit_facade import moved\nprint(moved.x)\n"
        );
        assert_eq!(
            apply_edits(&db, &edits, qualified_explicit_reexport),
            "import new_pkg.new as direct\nimport explicit_facade\nprint(direct.x, explicit_facade.moved.x)\n"
        );
    }

    #[test]
    fn namespace_move_only_rewrites_the_moved_portion() {
        let mut db = create_test_db(&[
            ("/one/ns/moved.py", "x = 1\n"),
            ("/two/ns/stays.py", "x = 2\n"),
            (
                "/consumer.py",
                "from ns import moved\nfrom ns import stays\nprint(moved.x, stays.x)\n",
            ),
            (
                "/aggregate.py",
                "import ns\nimport ns.moved as moved\nprint(ns.stays.x, moved.x)\n",
            ),
        ]);
        configure_search_paths(&mut db, vec!["/one".into(), "/two".into()]);
        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let aggregate = system_path_to_file(&db, "/aggregate.py").unwrap();
        let moved = system_path_to_file(&db, "/one/ns/moved.py").unwrap();
        let edits = will_rename_paths(
            &db,
            &[PathRename::directory(
                "/one/ns".into(),
                "/one/new_ns".into(),
            )],
            [consumer, aggregate, moved],
            |_| true,
        );
        assert_eq!(
            apply_edits(&db, &edits, consumer),
            "from new_ns import moved\nfrom ns import stays\nprint(moved.x, stays.x)\n"
        );
        assert_eq!(
            apply_edits(&db, &edits, aggregate),
            source_text(&db, aggregate).as_str()
        );
    }

    #[test]
    fn destination_binding_collision_does_not_cancel_local_edits() {
        let db = create_test_db(&[
            ("/old.py", "x = 1\n"),
            ("/consumer.py", "import old\nnew = 1\nprint(old.x, new)\n"),
        ]);
        assert_file_move(
            &db,
            "/old.py",
            "/new.py",
            "/consumer.py",
            "import new\nnew = 1\nprint(new.x, new)\n",
        );
    }

    fn will_rename(db: &dyn Db, renames: &[PathRename]) -> Vec<FileRenameEdit> {
        let project = db.project();
        let indexed_files = project.files(db);
        let open_files = project.open_files(db);
        will_rename_paths(
            db,
            renames,
            (&indexed_files)
                .into_iter()
                .chain(open_files.iter().copied()),
            |_| true,
        )
    }

    fn assert_file_move(db: &dyn Db, old_path: &str, new_path: &str, target: &str, expected: &str) {
        let edits = will_rename(db, &[PathRename::file(old_path.into(), new_path.into())]);
        let target = system_path_to_file(db, target).unwrap();
        assert_eq!(apply_edits(db, &edits, target), expected);
    }

    fn create_test_db(files: &[(&str, &str)]) -> TestDb {
        let mut db = TestDb::new(ProjectMetadata::new("test".into(), "/".into()));
        db.init_program_with_python_version(PythonVersion::latest_ty())
            .unwrap();
        for &(path, contents) in files {
            db.write_file(path, contents)
                .expect("write to memory file system to be successful");
        }
        db
    }

    fn configure_search_paths(db: &mut TestDb, src_roots: Vec<SystemPathBuf>) {
        let settings = SearchPathSettings::new(src_roots);
        let search_paths = settings
            .to_search_paths(db.system(), db.vendored(), &FallibleStrategy)
            .expect("valid search paths");
        Program::init_or_update(
            db,
            ProgramSettings {
                python_version: PythonVersionWithSource::default(),
                python_platform: PythonPlatform::default(),
                search_paths,
            },
        );
    }

    fn apply_edits(db: &dyn Db, edits: &[FileRenameEdit], file: File) -> String {
        let mut sorted_edits: Vec<_> = edits.iter().filter(|edit| edit.file == file).collect();
        sorted_edits.sort_by_key(|edit| std::cmp::Reverse(edit.range.start()));

        let mut result = source_text(db, file).as_str().to_owned();
        for edit in sorted_edits {
            let start = usize::from(edit.range.start());
            let end = usize::from(edit.range.end());
            result.replace_range(start..end, &edit.new_text);
        }
        result
    }
}
