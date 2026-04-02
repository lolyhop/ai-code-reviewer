from __future__ import annotations

import ast
import logging
import sys

logger = logging.getLogger(__name__)

# Source-root prefixes tried when resolving absolute imports.
# The empty string represents the repository root itself; "src" covers the
# common ``src/`` layout used by setuptools/poetry projects.
_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = ("", "src")

# Top-level names belonging to the standard library or the interpreter's
# built-in modules.  Imports whose first dotted component appears here are
# guaranteed not to resolve to files inside the repository, so we skip
# candidate generation for them entirely.
# ``sys.stdlib_module_names`` is available from Python 3.10 onward; we fall
# back gracefully to an empty set so the filter is simply a no-op on older
# interpreters.
_STDLIB_TOP_LEVEL: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", frozenset())
) | frozenset(sys.builtin_module_names)


def _module_parts_to_candidates(
    parts: list[str],
    source_roots: tuple[str, ...],
) -> list[str]:
    """Generate candidate repo-relative paths for an absolute module reference.

    For each source root and each target module, two candidates are produced:
    the plain ``module.py`` form and the ``module/__init__.py`` package form.

    Args:
        parts:
            Dotted module name split on ``"."``, e.g. ``["pkg", "sub", "mod"]``.
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
    ``file_path``.  A level of 1 means the current package; 2 means the parent
    package; and so on.

    Args:
        module:
            Dotted sub-module name after the dots, e.g. ``"utils"`` for
            ``from . import utils`` or ``"a.b"`` for ``from ..a import b``.
            ``None`` for a bare ``from . import name`` (each name is handled
            separately by the caller).
        level:
            Number of leading dots (must be >= 1).
        file_path:
            Repo-relative path of the file containing the import, using
            forward slashes (e.g. ``"pkg/sub/module.py"``).

    Returns:
        List of candidate repo-relative paths, or ``None`` if the relative
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
        A tuple ``(candidates, unresolvable)`` where ``unresolvable`` is
        ``True`` when the import cannot be mapped to candidate paths at all
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
        # dotted ``module_name`` was given.
        #
        # ``from . import utils``        → module_name=None  → anchor = pkg dir
        # ``from .utils import submod``  → module_name="utils" → anchor = utils
        #
        # In both cases ``alias.name`` may refer to a file under the anchor dir.
        if module_name:
            # anchor is the resolved pkg dir (strip ".py" from the first candidate)
            anchor_base = pkg_candidates[0].removesuffix(".py")
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
        # Absolute import: ``from pkg.mod import name``
        if not node.module:
            return [], False

        mod_parts = node.module.split(".")
        if mod_parts[0] in _STDLIB_TOP_LEVEL:
            return [], False

        # Candidates for the ``from`` module itself.
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
            Defaults to ``("", "src")``.

    Returns:
        A tuple ``(candidates, unresolvable_count)`` where:

        - ``candidates`` is the set of candidate repo-relative paths.
        - ``unresolvable_count`` is the number of import statements that could
          not be mapped to any candidate path (e.g. a relative import that
          escapes the repo root due to excessive dot levels).
    """
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as exc:
        logger.debug("SyntaxError parsing %s for import extraction: %s", file_path, exc)
        return set(), 0

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
