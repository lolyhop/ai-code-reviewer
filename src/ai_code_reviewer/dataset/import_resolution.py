from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass

from ai_code_reviewer.dataset.config import IMPORT_SOURCE_ROOTS

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE_ROOTS = IMPORT_SOURCE_ROOTS

# Top-level names belonging to the standard library or the interpreter's
# built-in modules.  Imports whose first dotted component appears here are
# guaranteed not to resolve to files inside the repository, so we skip
# candidate generation for them entirely.
# `sys.stdlib_module_names` is available from Python 3.10 onward; we fall
# back gracefully to an empty set so the filter is simply a no-op on older
# interpreters.
_STDLIB_TOP_LEVEL: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", frozenset())
) | frozenset(sys.builtin_module_names)


@dataclass(frozen=True)
class UsedImportFact:
    """Structured import fact for a locally-used imported binding.

    Attributes:
        import_kind:
            Statement kind: ``"import"`` or ``"from"``.
        module:
            Raw ``module`` from the import statement, when present.
        level:
            Relative import level (`0` for absolute imports).
        imported_name:
            Imported name from ``from ... import <name>`` statements.
            ``None`` for plain ``import ...`` statements.
        local_name:
            Local binding name introduced in the file namespace
            (alias when present, otherwise default bound name).
        candidate_paths:
            Candidate repo-relative paths that may define the imported binding.
            Includes both ``module.py`` and ``module/__init__.py`` forms.
    """

    import_kind: str
    module: str | None
    level: int
    imported_name: str | None
    local_name: str
    candidate_paths: frozenset[str]


def _module_parts_to_candidates(
    parts: list[str],
    source_roots: tuple[str, ...],
) -> list[str]:
    """Generate candidate repo-relative paths for an absolute module reference.

    For each source root and each target module, two candidates are produced:
    the plain `module.py` form and the `module/__init__.py` package form.

    Args:
        parts:
            Dotted module name split on `"."`, e.g. `["pkg", "sub", "mod"]`.
        source_roots:
            Source root prefixes to probe.  An empty string means repo root.

    Returns:
        Ordered list of candidate paths (duplicates may appear if roots overlap).
    """
    base = "/".join(parts)
    candidates: list[str] = []
    for root in source_roots:
        prefix = f"{root}/" if root else ""
        candidates.append(f"{prefix}{base}.py")
        candidates.append(f"{prefix}{base}/__init__.py")
    return candidates


def _relative_import_candidates(
    module: str | None,
    level: int,
    file_path: str,
) -> list[str] | None:
    """Compute candidate paths for a relative import statement.

    Relative imports are resolved with respect to the package directory of
    `file_path`.  A level of 1 means the current package; 2 means the parent
    package; and so on.

    Args:
        module:
            Dotted sub-module name after the dots, e.g. `"utils"` for
            `from . import utils` or `"a.b"` for `from ..a import b`.
            `None` for a bare `from . import name` (each name is handled
            separately by the caller).
        level:
            Number of leading dots (must be >= 1).
        file_path:
            Repo-relative path of the file containing the import, using
            forward slashes (e.g. `"pkg/sub/module.py"`).

    Returns:
        List of candidate repo-relative paths, or `None` if the relative
        level exceeds the available directory depth (would escape the repo root).
    """
    norm = file_path.replace("\\", "/").lstrip("/")
    dir_parts = norm.split("/")[:-1]

    # level=1 → same package directory; level=2 → parent; ...
    steps_up = level - 1
    if steps_up > len(dir_parts):
        return None

    anchor_parts = dir_parts[: len(dir_parts) - steps_up] if steps_up else dir_parts

    if module:
        target_parts = anchor_parts + module.split(".")
    else:
        target_parts = anchor_parts

    if not target_parts:
        return None

    base = "/".join(target_parts)
    return [f"{base}.py", f"{base}/__init__.py"]


def _candidates_for_node(
    node: ast.Import | ast.ImportFrom,
    file_path: str,
    source_roots: tuple[str, ...],
) -> tuple[list[str], bool]:
    """Derive candidate paths and an unresolved flag for a single AST import node.

    Args:
        node:
            An :class:`ast.Import` or :class:`ast.ImportFrom` node.
        file_path:
            Repo-relative path of the file containing the import.
        source_roots:
            Source-root prefixes for absolute import resolution.

    Returns:
        A tuple `(candidates, unresolvable)` where `unresolvable` is
        `True` when the import cannot be mapped to candidate paths at all
        (e.g. a relative import that escapes the repo root).
    """
    candidates: list[str] = []

    if isinstance(node, ast.Import):
        for alias in node.names:
            parts = alias.name.split(".")
            if parts[0] in _STDLIB_TOP_LEVEL:
                continue
            candidates.extend(_module_parts_to_candidates(parts, source_roots))
        return candidates, False

    # ast.ImportFrom
    level: int = node.level or 0

    if level > 0:
        # Relative import
        module_name: str | None = node.module  # e.g. "utils" or None

        # Candidates for the referenced package/module itself.
        pkg_candidates = _relative_import_candidates(module_name, level, file_path)
        if pkg_candidates is None:
            return [], True
        candidates.extend(pkg_candidates)

        # Each imported name could be a submodule regardless of whether a
        # dotted `module_name` was given.
        #
        # `from . import utils`        → module_name=None  → anchor = pkg dir
        # `from .utils import submod`  → module_name="utils" → anchor = utils
        #
        # In both cases `alias.name` may refer to a file under the anchor dir.
        if module_name:
            # anchor is the resolved pkg dir (strip ".py" from the first candidate)
            anchor_base: str | None = pkg_candidates[0].removesuffix(".py")
        else:
            anchor_candidates = _relative_import_candidates(None, level, file_path)
            anchor_base = (
                anchor_candidates[0].removesuffix(".py")
                if anchor_candidates is not None
                else None
            )

        if anchor_base is not None:
            for alias in node.names:
                if alias.name == "*":
                    continue
                sub = f"{anchor_base}/{alias.name}"
                candidates.append(f"{sub}.py")
                candidates.append(f"{sub}/__init__.py")
    else:
        # Absolute import: `from pkg.mod import name`
        if not node.module:
            return [], False

        mod_parts = node.module.split(".")
        if mod_parts[0] in _STDLIB_TOP_LEVEL:
            return [], False

        # Candidates for the `from` module itself.
        candidates.extend(_module_parts_to_candidates(mod_parts, source_roots))

        # Each imported name could be a submodule.
        for alias in node.names:
            if alias.name == "*":
                continue
            candidates.extend(
                _module_parts_to_candidates(mod_parts + [alias.name], source_roots)
            )

    return candidates, False


def resolve_import_candidates(
    source_code: str,
    file_path: str,
    source_roots: tuple[str, ...] = _DEFAULT_SOURCE_ROOTS,
) -> tuple[set[str], int]:
    """Parse Python source and return candidate repo-relative dependency paths.

    The returned candidate set contains paths that *may* exist in the repository
    and correspond to modules imported by the file.  Callers should probe
    these paths against the actual repository (e.g. via zipball lookup) to
    determine which ones really exist.

    Candidates are deduplicated.  The same path may satisfy multiple imports;
    this is intentional.

    Args:
        source_code:
            Full Python source text of the file to analyse.
        file_path:
            Repo-relative path of the file (used for relative-import anchoring).
        source_roots:
            Tuple of source-root prefixes to probe for absolute imports.
            Defaults to `("", "src")`.

    Returns:
        A tuple `(candidates, unresolvable_count)` where:

        - `candidates` is the set of candidate repo-relative paths.
        - `unresolvable_count` is the number of import statements that could
          not be mapped to any candidate path (e.g. a relative import that
          escapes the repo root due to excessive dot levels).
    """
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as exc:
        logger.debug("SyntaxError parsing %s for import extraction: %s", file_path, exc)
        # Treat an unparseable file as one unresolvable import statement so
        # callers see a non-zero count and can log the failure correctly.
        return set(), 1

    all_candidates: set[str] = set()
    unresolvable_count = 0

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        try:
            node_candidates, unresolvable = _candidates_for_node(
                node, file_path, source_roots
            )
        except Exception as exc:
            logger.debug(
                "Unexpected error resolving import in %s: %s", file_path, exc
            )
            unresolvable_count += 1
            continue

        if unresolvable:
            unresolvable_count += 1
        else:
            all_candidates.update(node_candidates)

    return all_candidates, unresolvable_count


def _collect_loaded_names(tree: ast.AST) -> frozenset[str]:
    """Collect all identifier names read from a module AST.

    Args:
        tree:
            Parsed module tree.

    Returns:
        Frozen set of names that appear in load context.
    """
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
    return frozenset(loaded)


def _attribute_chain_parts(node: ast.Attribute) -> tuple[str, tuple[str, ...]] | None:
    """Return ``(<root_name>, (<attr1>, ...))`` for a load-time attribute chain.

    Args:
        node:
            Attribute node to unwrap.

    Returns:
        Tuple of root identifier and ordered attribute suffix, or ``None`` when
        the attribute root is not a plain identifier.
    """
    attrs: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    if not attrs:
        return None
    return current.id, tuple(reversed(attrs))


def _collect_attribute_suffixes(
    tree: ast.AST,
) -> dict[str, frozenset[tuple[str, ...]]]:
    """Collect loaded attribute suffix chains keyed by root identifier name.

    Args:
        tree:
            Parsed module AST.

    Returns:
        Mapping ``root_name -> frozenset(suffix tuples)`` where each tuple is
        the ordered chain after the root name (e.g. ``("subpkg", "func")`` for
        ``pkg.subpkg.func``).
    """
    suffixes: dict[str, set[tuple[str, ...]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
            continue
        chain = _attribute_chain_parts(node)
        if chain is None:
            continue
        root_name, attr_suffix = chain
        suffixes.setdefault(root_name, set()).add(attr_suffix)
    return {name: frozenset(values) for name, values in suffixes.items()}


def _candidate_path_to_module_parts(candidate_path: str) -> list[str] | None:
    """Convert a candidate file path back to module dotted-part components.

    Args:
        candidate_path:
            Candidate path generated by import resolution.

    Returns:
        Module components derived from the path, or ``None`` for unsupported
        path shapes.
    """
    if candidate_path.endswith("/__init__.py"):
        module_path = candidate_path.removesuffix("/__init__.py")
    elif candidate_path.endswith(".py"):
        module_path = candidate_path.removesuffix(".py")
    else:
        return None
    parts = [part for part in module_path.split("/") if part]
    return parts or None


def _expand_candidates_with_attribute_usage(
    base_candidates: set[str],
    attribute_suffixes: frozenset[tuple[str, ...]],
    source_roots: tuple[str, ...],
) -> set[str]:
    """Expand module candidates by appending observed attribute-chain prefixes.

    Args:
        base_candidates:
            Candidate files that represent the locally bound imported object.
        attribute_suffixes:
            Attribute suffix chains observed for the local binding name.
        source_roots:
            Source-root prefixes used for candidate expansion.

    Returns:
        Additional candidate paths inferred from attribute usage.
    """
    if not base_candidates or not attribute_suffixes:
        return set()

    expanded: set[str] = set()
    for candidate in base_candidates:
        module_parts = _candidate_path_to_module_parts(candidate)
        if not module_parts:
            continue
        for suffix in attribute_suffixes:
            for depth in range(1, len(suffix) + 1):
                parts = module_parts + list(suffix[:depth])
                expanded.update(_module_parts_to_candidates(parts, source_roots))
    return expanded


def resolve_used_import_facts(
    source_code: str,
    file_path: str,
    source_roots: tuple[str, ...] = _DEFAULT_SOURCE_ROOTS,
) -> tuple[list[UsedImportFact], int]:
    """Return structured facts for imported bindings that are used in code.

    Args:
        source_code:
            Full source text to analyze.
        file_path:
            Repo-relative path of the analyzed file.
        source_roots:
            Source-root prefixes for absolute import resolution.

    Returns:
        A tuple ``(facts, unresolvable_count)`` where:

        - ``facts`` is an ordered list of :class:`UsedImportFact` entries.
        - ``unresolvable_count`` is the number of import statements that
          could not be mapped to candidate paths.
    """
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as exc:
        logger.debug("SyntaxError parsing %s for used-import extraction: %s", file_path, exc)
        return [], 1

    loaded_names = _collect_loaded_names(tree)
    attribute_suffixes = _collect_attribute_suffixes(tree)
    facts: list[UsedImportFact] = []
    unresolvable_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if not parts or parts[0] in _STDLIB_TOP_LEVEL:
                    continue
                local_name = alias.asname or parts[0]
                if local_name not in loaded_names:
                    continue
                base_candidates = set(_module_parts_to_candidates(parts, source_roots))
                usage_expanded = _expand_candidates_with_attribute_usage(
                    base_candidates,
                    attribute_suffixes.get(local_name, frozenset()),
                    source_roots,
                )
                candidates = frozenset(base_candidates | usage_expanded)
                facts.append(
                    UsedImportFact(
                        import_kind="import",
                        module=alias.name,
                        level=0,
                        imported_name=None,
                        local_name=local_name,
                        candidate_paths=candidates,
                    )
                )
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        level = node.level or 0
        try:
            module_candidates, unresolvable = _candidates_for_node(
                node, file_path, source_roots
            )
        except Exception as exc:
            logger.debug(
                "Unexpected error resolving used import in %s: %s", file_path, exc
            )
            unresolvable_count += 1
            continue
        if unresolvable:
            unresolvable_count += 1
            continue

        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            if local_name not in loaded_names:
                continue

            alias_candidates: set[str] = set(module_candidates)
            bound_candidates: set[str] = set()
            if level == 0 and node.module:
                mod_parts = node.module.split(".")
                if mod_parts and mod_parts[0] not in _STDLIB_TOP_LEVEL:
                    bound_candidates.update(
                        _module_parts_to_candidates(mod_parts + [alias.name], source_roots)
                    )
            elif level > 0:
                anchor_candidates = _relative_import_candidates(node.module, level, file_path)
                if anchor_candidates:
                    anchor_base = anchor_candidates[0].removesuffix(".py")
                    sub = f"{anchor_base}/{alias.name}"
                    bound_candidates.add(f"{sub}.py")
                    bound_candidates.add(f"{sub}/__init__.py")

            alias_candidates.update(bound_candidates)
            usage_expanded = _expand_candidates_with_attribute_usage(
                bound_candidates or alias_candidates,
                attribute_suffixes.get(local_name, frozenset()),
                source_roots,
            )
            alias_candidates.update(usage_expanded)

            facts.append(
                UsedImportFact(
                    import_kind="from",
                    module=node.module,
                    level=level,
                    imported_name=alias.name,
                    local_name=local_name,
                    candidate_paths=frozenset(alias_candidates),
                )
            )

    return facts, unresolvable_count


def _iter_module_level_import_nodes(
    module_tree: ast.Module,
) -> list[ast.Import | ast.ImportFrom]:
    """Return imports reachable at module level, including nested control flow.

    Traversal descends into module-level control-flow statement bodies (``if``,
    ``try``, ``with``, ``match``), but intentionally skips function/class
    scopes where imports do not define module re-exports.

    Args:
        module_tree:
            Parsed module AST.

    Returns:
        Ordered list of import nodes.
    """
    nodes: list[ast.Import | ast.ImportFrom] = []

    def _walk_block(statements: list[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                nodes.append(stmt)
                continue
            if isinstance(stmt, ast.If):
                _walk_block(stmt.body)
                _walk_block(stmt.orelse)
                continue
            if isinstance(stmt, ast.Try):
                _walk_block(stmt.body)
                _walk_block(stmt.orelse)
                _walk_block(stmt.finalbody)
                for handler in stmt.handlers:
                    _walk_block(handler.body)
                continue
            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                _walk_block(stmt.body)
                continue
            if isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    _walk_block(case.body)
                continue

    _walk_block(module_tree.body)
    return nodes


def resolve_reexport_candidates_for_symbol(
    source_code: str,
    file_path: str,
    exported_name: str,
    source_roots: tuple[str, ...] = _DEFAULT_SOURCE_ROOTS,
) -> tuple[set[str], int]:
    """Resolve candidate files that can provide `exported_name` from a module.

    This helper is intended for package ``__init__.py`` analysis where a name is
    re-exported via import statements (for example ``from .mod import Foo``).

    Args:
        source_code:
            Full source code of the exporting module.
        file_path:
            Repo-relative path of the exporting module.
        exported_name:
            Local name being imported by a caller.
        source_roots:
            Source-root prefixes for absolute import resolution.

    Returns:
        A tuple ``(candidates, unresolvable_count)``:

        - ``candidates`` are module candidate paths that can define
          ``exported_name``.
        - ``unresolvable_count`` counts import statements that could not be
          resolved to candidate paths.
    """
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as exc:
        logger.debug(
            "SyntaxError parsing %s for re-export analysis: %s", file_path, exc
        )
        return set(), 1

    candidates: set[str] = set()
    unresolvable_count = 0

    for node in _iter_module_level_import_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                if local_name != exported_name:
                    continue
                parts = alias.name.split(".")
                if parts and parts[0] not in _STDLIB_TOP_LEVEL:
                    candidates.update(_module_parts_to_candidates(parts, source_roots))
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            if local_name != exported_name:
                continue

            alias_node = ast.ImportFrom(
                module=node.module,
                names=[ast.alias(name=alias.name, asname=alias.asname)],
                level=node.level,
            )
            node_candidates, unresolvable = _candidates_for_node(
                alias_node, file_path, source_roots
            )
            if unresolvable:
                unresolvable_count += 1
            else:
                candidates.update(node_candidates)

    return candidates, unresolvable_count
