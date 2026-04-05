from __future__ import annotations

import asyncio
import io
import logging
import random
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import aiohttp
from tqdm import tqdm

import ai_code_reviewer.dataset.dataset_utils as dataset_utils
import ai_code_reviewer.dataset.http as http
import ai_code_reviewer.dataset.import_resolution as import_resolution
import ai_code_reviewer.dataset.patches as patches
import ai_code_reviewer.dataset.paths as path_utils
import ai_code_reviewer.dataset.symbol_extraction as symbol_extraction
from ai_code_reviewer.dataset import config


logger = logging.getLogger(__name__)


def _incoming_import_target_variants(
    changed_path: str,
    source_roots: tuple[str, ...] | None = None,
) -> frozenset[str]:
    """Return all path variants that the import resolver could emit for `changed_path`.

    The import resolver generates candidates in both the plain-module form
    (`pkg/mod.py`) and the package-init form (`pkg/mod/__init__.py`), and
    duplicates each under every configured source root.  This function produces
    the *inverse* set — the paths that would be generated for the import
    statement `import pkg.mod` — so that we can intersect it against the
    resolver output for a candidate file and confirm a real import link exists.

    Args:
        changed_path:
            Repo-relative POSIX path of the changed file
            (e.g. `"pkg/mod.py"` or `"src/pkg/mod/__init__.py"`).
        source_roots:
            Source-root prefixes to use for cross-applying path variants.
            When `None` the value from :data:`config.IMPORT_SOURCE_ROOTS`
            is used.  Pass a per-repo inferred tuple (from
            :func:`_infer_source_roots_from_zip_manifest`) for better accuracy.

    Returns:
        Frozen set of candidate path strings.  Empty if `changed_path` cannot
        be normalised (e.g. path traversal or empty string).
    """
    # Non-empty roots only: the empty-string (repo-root) form is excluded
    # because the direct normalised path already covers that case.
    effective_roots = (
        source_roots if source_roots is not None else config.IMPORT_SOURCE_ROOTS
    )
    non_empty_roots: tuple[str, ...] = tuple(r for r in effective_roots if r)

    norm = path_utils.normalize_repo_rel_path(changed_path)
    if norm is None:
        return frozenset()

    # Build the direct path and its module↔package counterpart.
    base_pair: list[str] = [norm]
    if norm.endswith("/__init__.py"):
        base_pair.append(norm.removesuffix("/__init__.py") + ".py")
    elif norm.endswith(".py"):
        base_pair.append(norm.removesuffix(".py") + "/__init__.py")

    # Cross-apply source-root prefixes: strip known roots (repo-root form) and
    # add them (src-layout form) so both sides of the layout are covered.
    variants: set[str] = set(base_pair)
    for p in base_pair:
        for root in non_empty_roots:
            prefix = f"{root}/"
            if p.startswith(prefix):
                variants.add(p[len(prefix) :])
            else:
                variants.add(f"{prefix}{p}")

    return frozenset(variants)


def _prune_intermediate_init_paths(paths: set[str]) -> tuple[set[str], int]:
    """Drop intermediate package ``__init__.py`` paths from outgoing candidates.

    A package ``__init__.py`` is considered intermediate when there is at least
    one resolved deeper dependency path under the same package directory.

    Args:
        paths:
            Candidate dependency paths.

    Returns:
        Tuple of ``(pruned_paths, removed_count)``.
    """
    if not paths:
        return set(), 0

    kept: set[str] = set(paths)
    removed = 0
    for candidate in list(paths):
        if not candidate.endswith("/__init__.py"):
            continue
        package_dir = candidate.removesuffix("/__init__.py")
        package_prefix = f"{package_dir}/"
        has_deeper_path = any(
            other != candidate and other.startswith(package_prefix) for other in paths
        )
        if has_deeper_path and candidate in kept:
            kept.remove(candidate)
            removed += 1
    return kept, removed


def _infer_zip_root_prefix(zf: zipfile.ZipFile) -> str | None:
    """Infer the GitHub archive root folder (first path segment) from a zipball.

    Args:
        zf:
            Open zip file.

    Returns:
        Root prefix ending with `/`, or None if the archive is empty.
    """
    counts: Counter[str] = Counter()
    for name in zf.namelist():
        if not name:
            continue
        parts = name.split("/")
        seg = parts[0]
        if not seg or seg.startswith("__MACOSX") or seg == ".DS_Store":
            continue
        if name.endswith("/") and len(parts) == 2 and parts[1] == "":
            counts[seg] += 1
            continue
        if not name.endswith("/"):
            counts[seg] += 1
    if not counts:
        return None
    root_seg, _ = counts.most_common(1)[0]
    return f"{root_seg}/"


def _build_repo_tree_string_from_zip_bytes(
    zip_bytes: bytes,
    root_label: str,
) -> str:
    """Build a deterministic tree-diagram string from zipball bytes.

    Args:
        zip_bytes:
            Raw zip bytes for a snapshot commit.
        root_label:
            Label for the rendered root directory.

    Returns:
        Directory tree diagram (similar to Unix `tree`) rooted at `root_label`,
        or an empty string when the archive cannot be parsed.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            root = _infer_zip_root_prefix(zf)
            if not root:
                return ""
            paths: list[str] = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if not name.startswith(root):
                    continue
                rel_path = path_utils.normalize_repo_rel_path(name[len(root) :])
                if rel_path is not None:
                    paths.append(rel_path)
    except zipfile.BadZipFile:
        return ""
    if not paths:
        return ""

    tree: dict[str, dict[str, Any]] = {}
    for path in sorted(set(paths)):
        node = tree
        for part in path.split("/"):
            node = node.setdefault(part, {})

    lines: list[str] = [f"{root_label}/"]

    def _render_subtree(node: dict[str, dict[str, Any]], prefix: str) -> None:
        items = sorted(
            node.items(),
            key=lambda kv: (0 if kv[1] else 1, kv[0].lower(), kv[0]),
        )
        for idx, (name, child) in enumerate(items):
            is_last = idx == len(items) - 1
            connector = "└── " if is_last else "├── "
            is_dir = bool(child)
            suffix = "/" if is_dir else ""
            lines.append(f"{prefix}{connector}{name}{suffix}")
            if is_dir:
                child_prefix = prefix + ("    " if is_last else "│   ")
                _render_subtree(child, child_prefix)

    _render_subtree(tree, "")
    return "\n".join(lines)


def _resolve_base_text_from_cache(
    base_cache: dict[str, str | None],
    path: str,
    cinfo: dict[str, Any],
) -> str | None:
    """Resolve base snapshot text for a path, including rename (old/new) lookups.

    Args:
        base_cache:
            Normalized path -> text.
        path:
            Repo-relative path from the comment (already normalized).
        cinfo:
            Compare entry for this file.

    Returns:
        File text at `base_commit`, or None if missing.
    """
    for key in (path, cinfo.get("previous_filename"), cinfo.get("filename")):
        if not key:
            continue
        nk = path_utils.normalize_repo_rel_path(key)
        if nk is None:
            continue
        t = base_cache.get(nk)
        if t is not None:
            return t
    return None


def _paths_for_base_zip_lookup(path_norm: str, cinfo: dict[str, Any]) -> set[str]:
    """Normalized repo paths to load from the base zipball.

    Args:
        path_norm:
            Normalized path from the dataset entry.
        cinfo:
            Compare file entry for this path.

    Returns:
        Repo-relative paths to request from the zipball at `base_commit`.
    """
    out: set[str] = set()
    for key in (path_norm, cinfo.get("previous_filename"), cinfo.get("filename")):
        if not key:
            continue
        nk = path_utils.normalize_repo_rel_path(key)
        if nk:
            out.add(nk)
    return out


def _extract_files_from_zip_bytes(
    zip_bytes: bytes,
    paths: set[str],
) -> dict[str, str | None]:
    """Read requested paths from a GitHub zipball in memory.

    Args:
        zip_bytes:
            Raw zip bytes.
        paths:
            Repo-relative paths.

    Returns:
        Mapping of each requested path to decoded text, or ``None`` if missing/unreadable.
    """
    out: dict[str, str | None] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            root = _infer_zip_root_prefix(zf)
            if not root:
                return {p: None for p in paths}
            for path in paths:
                member_path = path_utils.normalize_repo_rel_path(path)
                if member_path is None:
                    logger.warning("Skipping unsafe or empty zip path %r", path)
                    out[path] = None
                    continue
                member = f"{root}{member_path}"
                try:
                    info = zf.getinfo(member)
                except KeyError:
                    out[path] = None
                    continue
                if info.is_dir():
                    out[path] = None
                    continue
                if info.file_size > config.FILE_MAX_BYTES:
                    logger.warning(
                        "Skipping large file in zip %s (size=%s)",
                        path,
                        info.file_size,
                    )
                    out[path] = None
                    continue
                try:
                    raw = zf.read(member)
                    out[path] = raw.decode("utf-8")
                except (UnicodeDecodeError, ValueError) as exc:
                    logger.warning(
                        "Could not decode file %s from zip: %s",
                        path,
                        exc,
                    )
                    out[path] = None
    except zipfile.BadZipFile as exc:
        logger.error("Invalid zipball: %s", exc)
        return {p: None for p in paths}
    return out


def _scan_metadata_in_zip_bytes(
    zip_bytes: bytes,
    metadata_names: frozenset[str],
) -> dict[str, str]:
    """Scan a GitHub zipball and extract all files whose basename is in `metadata_names`.

    Unlike :func:`_extract_files_from_zip_bytes`, this function does not require
    knowing the paths in advance — it scans the full zip manifest and collects any
    entry whose final path segment matches a requested metadata filename.

    Args:
        zip_bytes:
            Raw zip bytes downloaded from the GitHub archive API.
        metadata_names:
            Set of filenames to match against the final path segment of each zip
            entry (e.g. `{"README.md", "requirements.txt"}`).

    Returns:
        Dict mapping each discovered repo-relative path to its decoded text content.
        Entries that exceed `config.FILE_MAX_BYTES` or cannot be decoded as UTF-8
        are silently skipped.
    """
    out: dict[str, str] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            root = _infer_zip_root_prefix(zf)
            if not root:
                return out
            for info in zf.infolist():
                name = info.filename
                if not name or info.is_dir():
                    continue
                basename = name.split("/")[-1]
                if basename not in metadata_names:
                    continue
                if info.file_size > config.FILE_MAX_BYTES:
                    logger.warning(
                        "Skipping large metadata file in zip %s (size=%s)",
                        name,
                        info.file_size,
                    )
                    continue
                # Strip the archive root prefix to get the repo-relative path.
                if not name.startswith(root):
                    continue
                repo_rel = name[len(root) :]
                repo_rel_norm = path_utils.normalize_repo_rel_path(repo_rel)
                if repo_rel_norm is None:
                    continue
                try:
                    raw = zf.read(info.filename)
                    out[repo_rel_norm] = raw.decode("utf-8")
                except (UnicodeDecodeError, ValueError) as exc:
                    logger.warning(
                        "Could not decode metadata file %s from zip: %s",
                        repo_rel_norm,
                        exc,
                    )
    except zipfile.BadZipFile as exc:
        logger.error("Invalid zipball while scanning metadata: %s", exc)
    return out


def _scan_symbol_usages_in_zip_bytes(
    zip_bytes: bytes,
    symbols: frozenset[str],
) -> dict[str, str | None]:
    """Scan all `.py` files in a zipball for word-boundary occurrences of any symbol.

    A single compiled alternation regex is used so each `.py` file in the
    archive is read exactly once regardless of how many symbols are searched.
    Files larger than :data:`config.FILE_MAX_BYTES` or that cannot be decoded
    as UTF-8 are silently skipped.

    Args:
        zip_bytes:
            Raw zip bytes downloaded from the GitHub archive API.
        symbols:
            Set of Python identifier names to search for.  Each name is
            matched as a whole word (`\\b` anchors) so partial matches inside
            longer identifiers are excluded.

    Returns:
        Dict mapping each discovered repo-relative `.py` path to its decoded
        text content, containing only files where at least one symbol was found.
        Returns an empty dict when `symbols` is empty or the zip is invalid.
    """
    if not symbols:
        return {}

    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in sorted(symbols)) + r")\b",
    )
    out: dict[str, str | None] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            root = _infer_zip_root_prefix(zf)
            if not root:
                return out
            for info in zf.infolist():
                name = info.filename
                if not name or info.is_dir():
                    continue
                if not name.endswith(".py"):
                    continue
                if not name.startswith(root):
                    continue
                repo_rel = name[len(root) :]
                repo_rel_norm = path_utils.normalize_repo_rel_path(repo_rel)
                if repo_rel_norm is None:
                    continue
                if info.file_size > config.FILE_MAX_BYTES:
                    logger.warning(
                        "Skipping large .py file in zip %s (size=%s) during symbol scan",
                        repo_rel_norm,
                        info.file_size,
                    )
                    continue
                try:
                    raw = zf.read(info.filename)
                    text = raw.decode("utf-8")
                except (UnicodeDecodeError, ValueError) as exc:
                    logger.debug(
                        "Could not decode .py file %s during symbol scan: %s",
                        repo_rel_norm,
                        exc,
                    )
                    continue
                if pattern.search(text):
                    out[repo_rel_norm] = text
    except zipfile.BadZipFile as exc:
        logger.error("Invalid zipball while scanning symbol usages: %s", exc)
    return out


def _infer_source_roots_from_zip_manifest(zip_bytes: bytes) -> tuple[str, ...]:
    """Infer Python source-root prefixes from the file-name manifest of a zipball.

    Reads only :meth:`zipfile.ZipFile.infolist` — no file content is decoded —
    so the operation is fast even for large archives.

    A top-level directory is considered a source root when it satisfies both:

    - It contains at least one Python package (a `__init__.py` at path depth
      ≥ 3 relative to the archive root, i.e. `top_dir/pkg/__init__.py`).
    - It is **not** itself a Python package (no `top_dir/__init__.py` exists).

    This heuristic correctly identifies `src/`, `lib/`, `app/`, etc.
    without hardcoding, while excluding top-level package directories like
    `my_package/` (which have their own `__init__.py`).

    The empty string (`""`) representing the repository root is always
    included in the returned tuple.

    Args:
        zip_bytes:
            Raw zip bytes downloaded from the GitHub archive API.

    Returns:
        Sorted tuple of source-root prefix strings, always containing `""`.
    """
    roots: set[str] = {""}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            archive_root = _infer_zip_root_prefix(zf)
            if not archive_root:
                return ("",)
            root_len = len(archive_root)
            # Collect repo-relative paths that end in __init__.py.
            top_dirs_with_packages: set[str] = set()
            top_package_dirs: set[str] = set()
            for info in zf.infolist():
                name = info.filename
                if info.is_dir() or not name.startswith(archive_root):
                    continue
                repo_rel = name[root_len:]
                if not repo_rel:
                    continue
                parts = repo_rel.split("/")
                if parts[-1] != "__init__.py":
                    continue
                depth = len(parts)  # e.g. ["src", "pkg", "__init__.py"] → 3
                if depth == 2:
                    # top_dir/__init__.py → top_dir is a package, not a source root
                    top_package_dirs.add(parts[0])
                elif depth >= 3:
                    # top_dir/pkg/__init__.py → top_dir contains a package
                    top_dirs_with_packages.add(parts[0])
            roots |= top_dirs_with_packages - top_package_dirs
    except zipfile.BadZipFile as exc:
        logger.warning("Could not infer source roots from zip manifest: %s", exc)
    return tuple(sorted(roots))


def _build_reexport_map(
    init_content_map: dict[str, str],
    source_roots: tuple[str, ...],
) -> dict[str, frozenset[str]]:
    """Build a map from sub-module candidate path → `__init__.py` files that import it.

    When a package's `__init__.py` re-exports a symbol from a sub-module
    (e.g. `from .mod import Foo`), callers that write `from pkg import Foo`
    import `pkg/__init__.py` rather than `pkg/mod.py` directly.  This map
    lets the incoming-dependency check detect that indirect import chain.

    Args:
        init_content_map:
            Mapping of repo-relative `__init__.py` path → file content,
            as extracted from the snapshot zipball for the changed file's
            enclosing packages.
        source_roots:
            Source-root prefixes to pass to :func:`resolve_import_candidates`
            when parsing each `__init__.py`.

    Returns:
        Dict mapping each sub-module candidate path (as produced by
        :func:`resolve_import_candidates`) to a frozenset of `__init__.py`
        repo-relative paths that contain an import resolving to that candidate.
        Returns an empty dict when `init_content_map` is empty.
    """
    reexport: dict[str, set[str]] = {}
    for init_path, content in init_content_map.items():
        if not init_path.endswith("/__init__.py"):
            continue
        candidates, _ = import_resolution.resolve_import_candidates(content, init_path, source_roots)
        for cand in candidates:
            reexport.setdefault(cand, set()).add(init_path)
    return {k: frozenset(v) for k, v in reexport.items()}


async def _resolve_outgoing_fact_paths(
    fact: import_resolution.UsedImportFact,
    source_roots: tuple[str, ...],
    get_content: Callable[[str], Awaitable[str | None]],
    reexport_cache: dict[tuple[str, str], frozenset[str]],
    max_depth: int,
) -> frozenset[str]:
    """Resolve concrete dependency paths for one used import fact.

    Args:
        fact:
            Structured import usage fact from
            :func:`resolve_used_import_facts`.
        source_roots:
            Source-root prefixes used by import resolution.
        get_content:
            Callback that returns raw file text by repo-relative path.
        reexport_cache:
            Memo cache keyed by ``(__init__.py path, exported symbol)``.
        max_depth:
            Maximum recursion depth for following re-export chains.

    Returns:
        Frozen set of resolved dependency paths.
    """
    start_paths = sorted(
        {
            p
            for p in (path_utils.normalize_repo_rel_path(path) for path in fact.candidate_paths)
            if p is not None and p.endswith(".py")
        },
    )
    if not start_paths:
        return frozenset()

    symbol = fact.imported_name if fact.import_kind == "from" else None
    queue: list[tuple[str, int]] = [(path, 0) for path in start_paths]
    visited: set[tuple[str, str | None]] = set()
    resolved: set[str] = set()

    while queue:
        current_path, depth = queue.pop(0)
        visit_key = (current_path, symbol)
        if visit_key in visited:
            continue
        visited.add(visit_key)

        current_content = await get_content(current_path)
        if current_content is None:
            continue

        if not current_path.endswith("/__init__.py") or symbol is None:
            resolved.add(current_path)
            continue

        if depth >= max_depth:
            resolved.add(current_path)
            continue

        cache_key = (current_path, symbol)
        next_paths = reexport_cache.get(cache_key)
        if next_paths is None:
            candidates, _ = import_resolution.resolve_reexport_candidates_for_symbol(
                current_content,
                current_path,
                symbol,
                source_roots,
            )
            next_norm: set[str] = {
                p
                for p in (
                    path_utils.normalize_repo_rel_path(candidate) for candidate in candidates
                )
                if p is not None and p.endswith(".py")
            }
            next_paths = frozenset(next_norm)
            reexport_cache[cache_key] = next_paths

        if not next_paths:
            resolved.add(current_path)
            continue

        queue.extend((nxt, depth + 1) for nxt in next_paths)

    return frozenset(resolved)


async def fetch_paths_from_zipball(
    owner: str,
    repo: str,
    base_commit: str,
    paths: set[str],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
) -> tuple[int, dict[str, str | None]]:
    """Download zip at `base_commit` and extract requested paths (one REST archive call).

    Args:
        owner:
            Repository owner login.
        repo:
            Repository name.
        base_commit:
            Commit SHA (base snapshot).
        paths:
            Repo-relative paths to read.
        session:
            The aiohttp client session.
        semaphore:
            Concurrency limiter for API calls.
        headers:
            Request headers.

    Returns:
        ``(status, {path: content | None})`` for each requested path.
    """
    url = f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/zipball/{base_commit}"
    status, data = await http.async_http_get_bytes(
        session,
        url,
        semaphore=semaphore,
        headers=headers,
        timeout=aiohttp.ClientTimeout(
            total=config.HTTP_ZIP_TIMEOUT_TOTAL,
            connect=config.HTTP_ZIP_TIMEOUT_CONNECT,
            sock_connect=config.HTTP_ZIP_TIMEOUT_SOCK_CONNECT,
            sock_read=config.HTTP_ZIP_TIMEOUT_SOCK_READ,
        ),
        max_response_bytes=config.ZIPBALL_MAX_BYTES,
    )
    if status != 200:
        text = data.decode("utf-8", errors="replace")[:400]
        logger.warning(
            "Zipball failed for %s/%s @ %s: HTTP %s %s",
            owner,
            repo,
            base_commit[:7],
            status,
            text,
        )
        return status, {}
    if not paths:
        return status, {}
    extracted = await asyncio.to_thread(_extract_files_from_zip_bytes, data, paths)
    return status, extracted


async def fetch_paths_and_metadata_from_zipball(
    owner: str,
    repo: str,
    commit: str,
    paths: set[str],
    metadata_names: frozenset[str],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
    symbols: frozenset[str] = frozenset(),
) -> tuple[int, dict[str, str | None], dict[str, str], dict[str, str | None]]:
    """Download a zipball once and extract explicit paths, metadata, and symbol usages.

    Downloads the archive at `commit` and performs up to three passes over the
    zip in a single thread invocation:

    1. Extract the caller-supplied `paths` (same semantics as
       :func:`fetch_paths_from_zipball`).
    2. Scan the full manifest for entries whose basename is in
       `metadata_names`.
    3. When `symbols` is non-empty, scan all `.py` files for word-boundary
       occurrences of any symbol via :func:`_scan_symbol_usages_in_zip_bytes`.

    Args:
        owner:
            Repository owner login.
        repo:
            Repository name.
        commit:
            Commit SHA to download.
        paths:
            Repo-relative paths to extract explicitly (may be empty).
        metadata_names:
            Basenames to search for anywhere in the archive (may be empty).
        session:
            The aiohttp client session.
        semaphore:
            Concurrency limiter for API calls.
        headers:
            Request headers.
        symbols:
            Python identifier names to search for across all `.py` files in
            the archive.  Pass an empty frozenset (default) to skip symbol
            scanning.

    Returns:
        A 4-tuple of:
            - HTTP status code
            - `{path: content | None}` for each entry in `paths`
            - `{repo_rel_path: content}` for each discovered metadata file
            - `{repo_rel_path: content}` for each `.py` file containing at
              least one of the requested symbols (empty dict when `symbols`
              is empty)
    """
    url = f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/zipball/{commit}"
    status, data = await http.async_http_get_bytes(
        session,
        url,
        semaphore=semaphore,
        headers=headers,
        timeout=aiohttp.ClientTimeout(
            total=config.HTTP_ZIP_TIMEOUT_TOTAL,
            connect=config.HTTP_ZIP_TIMEOUT_CONNECT,
            sock_connect=config.HTTP_ZIP_TIMEOUT_SOCK_CONNECT,
            sock_read=config.HTTP_ZIP_TIMEOUT_SOCK_READ,
        ),
        max_response_bytes=config.ZIPBALL_MAX_BYTES,
    )
    if status != 200:
        text = data.decode("utf-8", errors="replace")[:400]
        logger.warning(
            "Zipball failed for %s/%s @ %s: HTTP %s %s",
            owner,
            repo,
            commit[:7],
            status,
            text,
        )
        return status, {}, {}, {}

    def _extract_all() -> tuple[
        dict[str, str | None], dict[str, str], dict[str, str | None],
    ]:
        path_mapping = _extract_files_from_zip_bytes(data, paths) if paths else {}
        meta_mapping = (
            _scan_metadata_in_zip_bytes(data, metadata_names) if metadata_names else {}
        )
        usage_mapping = (
            _scan_symbol_usages_in_zip_bytes(data, symbols) if symbols else {}
        )
        return path_mapping, meta_mapping, usage_mapping

    path_mapping, meta_mapping, usage_mapping = await asyncio.to_thread(_extract_all)
    return status, path_mapping, meta_mapping, usage_mapping


async def fetch_compare_patches(
    session: aiohttp.ClientSession,
    owner: str,
    repo: str,
    base_commit: str,
    head_commit: str,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Fetch JSON compare between `base_commit` and `head_commit` and map paths to patch info.

    GitHub returns at most 300 entries in `files` on the first page; pagination does not
    append more file rows. Missing paths in the response cannot be recovered via JSON alone.

    Args:
        owner:
            Repository owner login.
        repo:
            Repository name.
        base_commit:
            Base of the comparison (merge-base side).
        head_commit:
            Head of the comparison (snapshot commit).
        session:
            The aiohttp client session.
        semaphore:
            Concurrency limiter for API calls.
        headers:
            Request headers.

    Returns:
        A tuple containing:
            - A dictionary mapping each relevant path (`filename` and
              `previous_filename` for renames) to patch metadata: keys
              ``patch``, ``status``, ``filename``, ``previous_filename``
              (each ``str | None``).
            - A boolean indicating whether the compare returned exactly 300
              files (possible truncation per GitHub).
    """
    url = (
        f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/compare/"
        f"{base_commit}...{head_commit}"
    )
    status, data = await http.async_http_get_json(
        session,
        url,
        semaphore=semaphore,
        headers=headers,
        timeout=aiohttp.ClientTimeout(
            total=config.HTTP_JSON_TIMEOUT_TOTAL,
            connect=config.HTTP_JSON_TIMEOUT_CONNECT,
            sock_connect=config.HTTP_JSON_TIMEOUT_SOCK_CONNECT,
            sock_read=config.HTTP_JSON_TIMEOUT_SOCK_READ,
        ),
    )
    if status == 404:
        logger.warning(
            "Compare not found for %s/%s %s...%s",
            owner,
            repo,
            base_commit[:7],
            head_commit[:7],
        )
        return {}, False
    if status != 200:
        logger.error(
            "Compare failed %s/%s: HTTP %s",
            owner,
            repo,
            status,
        )
        return {}, False
    files = (data or {}).get("files") or []
    maybe_truncated = len(files) == 300
    if maybe_truncated:
        logger.warning(
            "Compare returned 300 files for %s/%s %s...%s — JSON file list may be truncated.",
            owner,
            repo,
            base_commit[:7],
            head_commit[:7],
        )

    path_to_info: dict[str, dict[str, Any]] = {}
    for entry in files:
        fn = entry.get("filename")
        prev = entry.get("previous_filename")
        info = {
            "patch": entry.get("patch"),
            "status": entry.get("status"),
            "filename": fn,
            "previous_filename": prev,
        }
        if fn:
            nfn = path_utils.normalize_repo_rel_path(fn)
            if nfn:
                path_to_info[nfn] = info
        if prev:
            nprev = path_utils.normalize_repo_rel_path(prev)
            if nprev:
                path_to_info[nprev] = info
    return path_to_info, maybe_truncated


_COMPARE_QUEUE_STOP = object()
_REPO_ENRICH_QUEUE_STOP = object()


async def gather_compare_patches_bounded(
    session: aiohttp.ClientSession,
    owner: str,
    repo: str,
    compare_metas: list[tuple[str, str, str]],
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
) -> list[tuple[dict[str, dict[str, Any]], bool] | BaseException]:
    """Run `fetch_compare_patches` for each meta with a bounded worker pool.

    Args:
        session:
            Shared aiohttp session.
        owner:
            Repository owner login.
        repo:
            Repository name.
        compare_metas:
            Ordered `(pr_number, base_commit, snapshot_commit)` tuples.
        semaphore:
            API concurrency limiter.
        headers:
            GitHub REST headers.

    Returns:
        One list entry per `compare_metas` element, in order: either result
        `(path_to_info, maybe_truncated)` or `BaseException` if the fetch failed.
    """
    n = len(compare_metas)
    if n == 0:
        return []
    results: list[tuple[dict[str, dict[str, Any]], bool] | BaseException | None] = [
        None,
    ] * n
    worker_count = min(n, config.GITHUB_API_CONCURRENCY)
    work_queue: asyncio.Queue[tuple[int, tuple[str, str, str]] | object] = (
        asyncio.Queue()
    )
    for i, meta in enumerate(compare_metas):
        await work_queue.put((i, meta))
    for _ in range(worker_count):
        await work_queue.put(_COMPARE_QUEUE_STOP)

    async def _compare_worker() -> None:
        while True:
            item = await work_queue.get()
            if item is _COMPARE_QUEUE_STOP:
                break
            idx, (pr_number, base_commit, snapshot_commit) = item
            try:
                results[idx] = await fetch_compare_patches(
                    session,
                    owner,
                    repo,
                    base_commit,
                    snapshot_commit,
                    semaphore,
                    headers,
                )
            except BaseException as exc:
                results[idx] = exc

    await asyncio.gather(*(_compare_worker() for _ in range(worker_count)))
    out: list[tuple[dict[str, dict[str, Any]], bool] | BaseException] = []
    for r in results:
        if r is None:
            raise RuntimeError("gather_compare_patches_bounded: missing result slot")
        out.append(r)
    return out


def _augment_snapshot_with_no_comment_files(
    path_map: MutableMapping[str, Any],
    cmap: dict[str, dict[str, Any]],
    rng: random.Random,
) -> None:
    """Add changed .py files without comments to `path_map`, balanced per snapshot.

    Args:
        path_map:
            Mapping of normalized path → file entry for one snapshot commit.  Mutated
            in place.
        cmap:
            Compare result for the same `(pr_number, snapshot_commit)`: normalized
            path → `{"patch", "status", "filename", "previous_filename"}`.
        rng:
            Seeded :class:`random.Random` instance shared across all snapshots in a
            repo enrichment call so the seed is applied consistently.
    """
    commented_count = sum(1 for fe in path_map.values() if fe.get("comments"))
    if commented_count == 0:
        return

    candidates: list[str] = [
        path_norm
        for path_norm, cinfo in cmap.items()
        if (
            path_norm.endswith(".py")
            and cinfo.get("patch")
            and path_norm not in path_map
        )
    ]
    if not candidates:
        return

    candidates.sort()
    cap = min(commented_count, len(candidates))
    selected = rng.sample(candidates, cap)
    for path_norm in selected:
        path_map[path_norm] = {"comments": []}
    logger.debug(
        "Augmented snapshot with %d no-comment .py file(s) (cap=%d, eligible=%d)",
        len(selected),
        cap,
        len(candidates),
    )


async def _enrich_one_repo(
    dataset: MutableMapping[str, Any],
    repo_name: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
) -> None:
    """Enrich a single top-level repo key (compare, zipball, patch fields).

    Args:
        dataset:
            Dataset to enrich.
        repo_name:
            Repository name.
        session:
            The aiohttp client session.
        semaphore:
            API concurrency limiter.
        headers:
            GitHub REST headers.
    """
    parts = repo_name.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        logger.warning("Invalid repo_name %r; dropping repo.", repo_name)
        del dataset[repo_name]
        return
    owner, repo = parts[0], parts[1]

    for pr_number in list(dataset[repo_name].keys()):
        pr_entry = dataset[repo_name][pr_number]
        base_commit = pr_entry.get("base_commit")
        if not base_commit:
            logger.warning(
                "No base_commit for %s PR %s; dropping PR.",
                repo_name,
                pr_number,
            )
            del dataset[repo_name][pr_number]
            continue

    compare_metas: list[tuple[str, str, str]] = []
    for pr_number in list(dataset[repo_name].keys()):
        pr_entry = dataset[repo_name][pr_number]
        base_commit = pr_entry.get("base_commit")
        if not base_commit:
            continue
        compare_metas.extend(
            (pr_number, base_commit, sc) for sc in pr_entry["commits"]
        )

    compare_results: list[tuple[str, str, str, dict[str, dict[str, Any]], bool]] = []
    if compare_metas:
        compare_raw = await gather_compare_patches_bounded(
            session,
            owner,
            repo,
            compare_metas,
            semaphore,
            headers,
        )
        for meta, result in zip(compare_metas, compare_raw):
            pr_number, base_commit, snapshot_commit = meta
            if isinstance(result, BaseException):
                logger.error(
                    "Compare task failed for %s PR %s snapshot %s: %s",
                    repo_name,
                    pr_number,
                    snapshot_commit[:7],
                    result,
                )
                cmap: dict[str, dict[str, Any]] = {}
                trunc = False
            else:
                cmap, trunc = result
            compare_results.append(
                (pr_number, base_commit, snapshot_commit, cmap, trunc),
            )

    compare_cache: dict[tuple[Any, str], dict[str, dict[str, Any]]] = {}
    for pr_number, _base_commit, snapshot_commit, cmap, _trunc in compare_results:
        compare_cache[(pr_number, snapshot_commit)] = cmap

    if config.INCLUDE_NO_COMMENT_FILES:
        rng = random.Random(config.SEED)
        for pr_number in list(dataset[repo_name].keys()):
            pr_entry = dataset[repo_name][pr_number]
            for snapshot_commit, path_map in pr_entry["commits"].items():
                cmap = compare_cache.get((pr_number, snapshot_commit), {})
                _augment_snapshot_with_no_comment_files(path_map, cmap, rng)

    # Collect which Python files need to be fetched from each base-commit zipball.
    base_to_paths: defaultdict[str, set[str]] = defaultdict(set)
    # All unique base commits (superset of base_to_paths keys — includes PRs whose
    # Python files are all "added" so they don't need base content, but whose
    # metadata files may still have changed and need a base-side diff).
    all_base_commits: set[str] = set()
    for pr_number in list(dataset[repo_name].keys()):
        pr_entry = dataset[repo_name][pr_number]
        base_commit = pr_entry.get("base_commit")
        if not base_commit:
            continue
        all_base_commits.add(base_commit)
        for snapshot_commit, path_map in pr_entry["commits"].items():
            cmap = compare_cache.get((pr_number, snapshot_commit), {})
            for path in path_map:
                if path in {"metadata_files", "file_tree"}:
                    continue
                path_norm = path_utils.normalize_repo_rel_path(path)
                if path_norm is None:
                    continue
                cinfo = cmap.get(path_norm)
                if not cinfo or not cinfo.get("patch"):
                    continue
                file_status = (cinfo.get("status") or "").lower()
                if file_status == "added":
                    continue
                for nk in _paths_for_base_zip_lookup(path_norm, cinfo):
                    base_to_paths[base_commit].add(nk)

    _metadata_names: frozenset[str] = frozenset(config.METADATA_FILE_NAMES)

    # base_commit → {path: content | None}  (Python file base content)
    base_caches: dict[str, dict[str, str | None]] = {}
    # base_commit → {repo_rel_path: content}  (metadata file base content)
    base_metadata_caches: dict[str, dict[str, str]] = {}

    async def fetch_one_base(
        base_commit: str,
    ) -> tuple[str, int, dict[str, str | None], dict[str, str]]:
        paths_needed = base_to_paths.get(base_commit, set())
        (
            status,
            path_mapping,
            meta_mapping,
            _usage,
        ) = await fetch_paths_and_metadata_from_zipball(
            owner,
            repo,
            base_commit,
            paths_needed,
            _metadata_names,
            session,
            semaphore,
            headers,
        )
        return base_commit, status, path_mapping, meta_mapping

    all_base_list = list(all_base_commits)
    zip_tasks = [fetch_one_base(bs) for bs in all_base_list]
    zip_results: list[tuple[str, int, dict[str, str | None], dict[str, str]]] = []
    if zip_tasks:
        zip_raw = await asyncio.gather(*zip_tasks, return_exceptions=True)
        for bs, res in zip(all_base_list, zip_raw):
            if isinstance(res, BaseException):
                logger.error(
                    "Zipball task failed for %s @ %s: %s",
                    repo_name,
                    bs[:7],
                    res,
                )
                zip_results.append((bs, 599, {}, {}))
            else:
                zip_results.append(res)

    for base_commit, status, path_map_result, meta_map_result in zip_results:
        if status != 200:
            if base_to_paths.get(base_commit):
                # Only drop PRs when we actually needed Python base content.
                logger.warning(
                    "Zipball failed for %s @ %s: HTTP %s; dropping PRs with this base.",
                    repo_name,
                    base_commit[:7],
                    status,
                )
                for pr_number in list(dataset[repo_name].keys()):
                    if dataset[repo_name][pr_number].get("base_commit") == base_commit:
                        del dataset[repo_name][pr_number]
            else:
                logger.warning(
                    "Base zipball (metadata-only) failed for %s @ %s: HTTP %s; "
                    "metadata diffs will be skipped.",
                    repo_name,
                    base_commit[:7],
                    status,
                )
        else:
            base_caches[base_commit] = path_map_result
            base_metadata_caches[base_commit] = meta_map_result

    for pr_number in list(dataset[repo_name].keys()):
        pr_entry = dataset[repo_name][pr_number]
        base_commit = pr_entry.get("base_commit")
        if not base_commit:
            continue
        base_cache = base_caches.get(base_commit, {})

        snapshot_commits = list(pr_entry["commits"].keys())

        for snapshot_commit in snapshot_commits:
            cmap = compare_cache.get((pr_number, snapshot_commit), {})
            for path in list(pr_entry["commits"][snapshot_commit].keys()):
                if path in {"metadata_files", "file_tree"}:
                    continue
                file_entry = pr_entry["commits"][snapshot_commit][path]
                path_norm = path_utils.normalize_repo_rel_path(path)
                if path_norm is None:
                    del pr_entry["commits"][snapshot_commit][path]
                    continue
                cinfo = cmap.get(path_norm)
                if not cinfo:
                    del pr_entry["commits"][snapshot_commit][path]
                    continue
                patch = cinfo.get("patch")
                if not patch:
                    del pr_entry["commits"][snapshot_commit][path]
                    continue
                file_status = (cinfo.get("status") or "").lower()
                if file_status == "added":
                    file_entry["base_content"] = ""
                    file_entry["patch"] = patch
                else:
                    base_text = _resolve_base_text_from_cache(
                        base_cache, path_norm, cinfo,
                    )
                    if base_text is None:
                        del pr_entry["commits"][snapshot_commit][path]
                        continue
                    file_entry["base_content"] = base_text
                    file_entry["patch"] = patch

    # Pre-Phase 4: prefetch snapshot zipballs and infer per-commit source roots.
    # Raw bytes are cached here to avoid a second download in Phase 6.

    # Collect the full set of snapshot commits across every PR.
    all_snapshot_commits: set[str] = set()
    for _pr_number in list(dataset[repo_name].keys()):
        _pr_entry = dataset[repo_name][_pr_number]
        all_snapshot_commits.update(_pr_entry["commits"].keys())

    # snapshot_commit → raw zip bytes (populated here, consumed in Phase 6)
    commit_to_zip_bytes: dict[str, bytes] = {}
    # snapshot_commit → inferred source-root prefixes (e.g. ("", "src"))
    commit_to_source_roots: dict[str, tuple[str, ...]] = {}
    # snapshot_commit → newline-separated sorted repo file tree at head
    commit_to_file_tree: dict[str, str] = {}

    async def _prefetch_snapshot_zip(
        snapshot_commit: str,
    ) -> tuple[str, int, bytes]:
        url = f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/zipball/{snapshot_commit}"
        status, data = await http.async_http_get_bytes(
            session,
            url,
            semaphore=semaphore,
            headers=headers,
            timeout=aiohttp.ClientTimeout(
                total=config.HTTP_ZIP_TIMEOUT_TOTAL,
                connect=config.HTTP_ZIP_TIMEOUT_CONNECT,
                sock_connect=config.HTTP_ZIP_TIMEOUT_SOCK_CONNECT,
                sock_read=config.HTTP_ZIP_TIMEOUT_SOCK_READ,
            ),
            max_response_bytes=config.ZIPBALL_MAX_BYTES,
        )
        return snapshot_commit, status, data

    prefetch_raw = await asyncio.gather(
        *[_prefetch_snapshot_zip(c) for c in all_snapshot_commits],
        return_exceptions=True,
    )
    for _prefetch_res in prefetch_raw:
        if isinstance(_prefetch_res, BaseException):
            logger.error(
                "Snapshot zip prefetch failed for %s: %s",
                repo_name,
                _prefetch_res,
            )
        else:
            _sc, _pf_status, _pf_data = _prefetch_res
            if _pf_status == 200:
                commit_to_zip_bytes[_sc] = _pf_data
                commit_to_source_roots[_sc] = await asyncio.to_thread(
                    _infer_source_roots_from_zip_manifest, _pf_data,
                )
                commit_to_file_tree[_sc] = await asyncio.to_thread(
                    _build_repo_tree_string_from_zip_bytes, _pf_data, repo,
                )
            else:
                logger.warning(
                    "Snapshot zip prefetch HTTP %s for %s @ %s; "
                    "deps and metadata will be empty for this commit.",
                    _pf_status,
                    repo_name,
                    _sc[:7],
                )

    for _pr_number in list(dataset[repo_name].keys()):
        _pr_entry = dataset[repo_name][_pr_number]
        for _snapshot_commit, _path_map in _pr_entry["commits"].items():
            _tree_string = commit_to_file_tree.get(_snapshot_commit)
            if _tree_string:
                _path_map["file_tree"] = {"tree": _tree_string}

    # Phase 4+5: apply patches inline; collect outgoing dep candidates and
    # changed symbols for incoming-dependency search.
    # Candidates/symbols are stored on file_entry under temporary keys
    # ("_dep_candidates", "_changed_symbols") to avoid (commit, path) collisions
    # across PRs; commit-level unions are collected in the dicts below.
    commit_to_dep_candidates: defaultdict[str, set[str]] = defaultdict(set)
    commit_to_changed_symbols: defaultdict[str, set[str]] = defaultdict(set)
    # __init__.py paths for all enclosing packages of changed .py files;
    # used in Phase 6 to build the re-export map for Phase 7.
    commit_to_init_paths: defaultdict[str, set[str]] = defaultdict(set)

    def _phase45_process_snapshot(
        snapshot_commit: str,
        path_map: MutableMapping[str, Any],
    ) -> tuple[str, set[str], set[str], set[str]]:
        _commit_source_roots = commit_to_source_roots.get(
            snapshot_commit, config.IMPORT_SOURCE_ROOTS,
        )
        snapshot_dep_candidates: set[str] = set()
        snapshot_changed_symbols: set[str] = set()
        snapshot_init_paths: set[str] = set()

        for path, file_entry in path_map.items():
            if path in {"metadata_files", "file_tree"}:
                continue
            base_str: str = file_entry.get("base_content") or ""
            patch_str: str = file_entry.get("patch") or ""
            if not patch_str:
                continue

            result: patches.PatchedContentResult | None = patches.compute_patched_content(
                base_str, patch_str, path,
            )
            if result is None:
                for comment in file_entry.get("comments", []):
                    comment["annotated_start_line"] = None
                    comment["annotated_end_line"] = None
                continue

            file_entry["patched_content"] = result.annotated
            file_entry["_head_text"] = result.head_text
            for comment in file_entry.get("comments", []):
                a_s, a_e = patches.head_blob_range_to_annotated_1based(
                    comment.get("head_start_line"),
                    comment.get("head_end_line"),
                    result.head_to_annotated,
                )
                comment["annotated_start_line"] = a_s
                comment["annotated_end_line"] = a_e

            if not path.endswith(".py"):
                continue

            used_facts, used_unresolvable = import_resolution.resolve_used_import_facts(
                result.head_text, path, _commit_source_roots,
            )
            if used_unresolvable:
                logger.debug(
                    "Unresolvable used-import statements in %s: %d",
                    path,
                    used_unresolvable,
                )
            used_candidates = {
                cand
                for fact in used_facts
                for cand in fact.candidate_paths
                if cand.endswith(".py") and cand != path
            }
            if used_facts:
                file_entry["_dep_usage_facts"] = used_facts
            if used_candidates:
                file_entry["_dep_candidates"] = used_candidates
                snapshot_dep_candidates |= used_candidates
            elif used_unresolvable > 0:
                broad_candidates, unresolvable = import_resolution.resolve_import_candidates(
                    result.head_text, path, _commit_source_roots,
                )
                if unresolvable:
                    logger.debug(
                        "Unresolvable fallback import statements in %s: %d",
                        path,
                        unresolvable,
                    )
                broad_candidates = {
                    c for c in broad_candidates if c.endswith(".py") and c != path
                }
                if broad_candidates:
                    file_entry["_dep_candidates"] = broad_candidates
                    snapshot_dep_candidates |= broad_candidates

            changed_lines = symbol_extraction.extract_patch_head_line_numbers(patch_str)
            raw_symbols = symbol_extraction.extract_changed_symbols(
                result.head_text,
                changed_lines,
                min_name_length=config.INCOMING_DEP_MIN_SYMBOL_LENGTH,
            )
            if raw_symbols:
                capped: frozenset[str] = frozenset(
                    sorted(raw_symbols, key=len, reverse=True)[
                        : config.INCOMING_DEP_MAX_SYMBOLS_PER_FILE
                    ],
                )
                file_entry["_changed_symbols"] = capped
                snapshot_changed_symbols |= set(capped)

            _path_parts = path.rstrip("/").split("/")[:-1]
            for _depth in range(1, len(_path_parts) + 1):
                _init_path = "/".join(_path_parts[:_depth]) + "/__init__.py"
                snapshot_init_paths.add(_init_path)

        return (
            snapshot_commit,
            snapshot_dep_candidates,
            snapshot_changed_symbols,
            snapshot_init_paths,
        )

    phase45_tasks = [
        asyncio.to_thread(_phase45_process_snapshot, snapshot_commit, path_map)
        for pr_number in list(dataset[repo_name].keys())
        for snapshot_commit, path_map in dataset[repo_name][pr_number][
            "commits"
        ].items()
    ]
    phase45_results = await asyncio.gather(*phase45_tasks)
    for (
        snapshot_commit,
        snapshot_dep_candidates,
        snapshot_changed_symbols,
        snapshot_init_paths,
    ) in phase45_results:
        if snapshot_dep_candidates:
            commit_to_dep_candidates[snapshot_commit] |= snapshot_dep_candidates
        if snapshot_changed_symbols:
            commit_to_changed_symbols[snapshot_commit] |= snapshot_changed_symbols
        if snapshot_init_paths:
            commit_to_init_paths[snapshot_commit] |= snapshot_init_paths

    # Phase 6: extract content from pre-fetched snapshot zipballs.
    # Keep zip bytes available for lazy on-demand extraction in Phase 7 when
    # recursive re-export resolution discovers additional paths.
    dep_commit_to_content: dict[str, dict[str, str | None]] = {}
    snapshot_metadata_head: dict[str, dict[str, str]] = {}
    dep_commit_to_usage: dict[str, dict[str, str | None]] = {}
    commit_to_init_content: dict[str, dict[str, str]] = {}

    def _make_zip_extractor(
        snapshot_commit: str,
    ) -> tuple[str, Any]:
        zip_data = commit_to_zip_bytes.get(snapshot_commit)
        paths_needed = commit_to_dep_candidates.get(snapshot_commit, set())
        snapshot_symbols = frozenset(
            commit_to_changed_symbols.get(snapshot_commit, set()),
        )
        init_paths = commit_to_init_paths.get(snapshot_commit, set())

        def _extract() -> tuple[
            dict[str, str | None],
            dict[str, str],
            dict[str, str | None],
            dict[str, str],
        ]:
            if zip_data is None:
                return {}, {}, {}, {}
            path_mapping = (
                _extract_files_from_zip_bytes(zip_data, paths_needed)
                if paths_needed
                else {}
            )
            meta_mapping = (
                _scan_metadata_in_zip_bytes(zip_data, _metadata_names)
                if _metadata_names
                else {}
            )
            usage_mapping = (
                _scan_symbol_usages_in_zip_bytes(zip_data, snapshot_symbols)
                if snapshot_symbols
                else {}
            )
            # Extract __init__.py files for the re-export map; filter out
            # None values (missing files produce None from the extractor).
            init_raw = (
                _extract_files_from_zip_bytes(zip_data, init_paths)
                if init_paths
                else {}
            )
            init_mapping: dict[str, str] = {
                p: c for p, c in init_raw.items() if c is not None
            }
            return path_mapping, meta_mapping, usage_mapping, init_mapping

        return snapshot_commit, _extract

    all_snapshot_list = list(all_snapshot_commits)
    extractors = [_make_zip_extractor(c) for c in all_snapshot_list]
    dep_raw = await asyncio.gather(
        *[asyncio.to_thread(fn) for _, fn in extractors],
        return_exceptions=True,
    )

    for (snapshot_commit, _), raw_res in zip(extractors, dep_raw):
        if isinstance(raw_res, BaseException):
            logger.error(
                "Snapshot zip extraction failed for %s @ %s: %s",
                repo_name,
                snapshot_commit[:7],
                raw_res,
            )
            dep_commit_to_content[snapshot_commit] = {}
            snapshot_metadata_head[snapshot_commit] = {}
            dep_commit_to_usage[snapshot_commit] = {}
            commit_to_init_content[snapshot_commit] = {}
        else:
            path_mapping, meta_mapping, usage_mapping, init_mapping = raw_res
            dep_commit_to_content[snapshot_commit] = path_mapping
            snapshot_metadata_head[snapshot_commit] = meta_mapping
            dep_commit_to_usage[snapshot_commit] = usage_mapping
            commit_to_init_content[snapshot_commit] = init_mapping

    # Phase 7: attach outgoing_dependencies and incoming_dependencies.

    for pr_number in list(dataset[repo_name].keys()):
        pr_entry = dataset[repo_name][pr_number]
        for snapshot_commit, path_map in pr_entry["commits"].items():
            content_map = dep_commit_to_content.get(snapshot_commit, {})
            usage_map = dep_commit_to_usage.get(snapshot_commit, {})
            snapshot_zip_bytes = commit_to_zip_bytes.get(snapshot_commit)
            # Use per-commit inferred source roots for both outgoing candidate
            # filtering and incoming dep import-link validation.
            snapshot_source_roots = commit_to_source_roots.get(
                snapshot_commit, config.IMPORT_SOURCE_ROOTS,
            )
            # Build a one-hop re-export map: sub-module path →
            # frozenset of __init__.py paths that import from it.
            # Used to detect callers that import a changed symbol via an
            # intermediate package __init__.py re-export.
            reexport_map = _build_reexport_map(
                commit_to_init_content.get(snapshot_commit, {}),
                snapshot_source_roots,
            )
            outgoing_imports_analyzed = 0
            outgoing_reexport_resolved = 0
            outgoing_resolution_fallbacks = 0
            outgoing_init_pruned = 0
            reexport_cache: dict[tuple[str, str], frozenset[str]] = {}

            async def _get_raw_dep_content(
                dep_path: str,
                *,
                _path_map: dict[str, Any] = path_map,
                _content_map: dict[str, str | None] = content_map,
                _snapshot_zip_bytes: bytes | None = snapshot_zip_bytes,
            ) -> str | None:
                dep_norm = path_utils.normalize_repo_rel_path(dep_path)
                if dep_norm is None:
                    return None

                dep_entry = _path_map.get(dep_norm)
                if dep_entry is not None:
                    head_text = dep_entry.get("_head_text")
                    if isinstance(head_text, str):
                        return head_text

                if dep_norm in _content_map:
                    return _content_map[dep_norm]

                if _snapshot_zip_bytes is None:
                    _content_map[dep_norm] = None
                    return None

                extracted = await asyncio.to_thread(
                    _extract_files_from_zip_bytes,
                    _snapshot_zip_bytes,
                    {dep_norm},
                )
                value = extracted.get(dep_norm)
                _content_map[dep_norm] = value
                return value

            for path, file_entry in path_map.items():
                if path in {"metadata_files", "file_tree"}:
                    continue

                # ---- outgoing dependencies ---- #
                candidates = file_entry.pop("_dep_candidates", None)
                usage_facts = file_entry.pop("_dep_usage_facts", None)
                candidate_set: set[str] = set()
                if usage_facts:
                    for fact in usage_facts:
                        outgoing_imports_analyzed += 1
                        resolved = await _resolve_outgoing_fact_paths(
                            fact,
                            snapshot_source_roots,
                            _get_raw_dep_content,
                            reexport_cache,
                            config.OUTGOING_REEXPORT_MAX_DEPTH,
                        )
                        if resolved:
                            if fact.imported_name is not None and any(
                                not p.endswith("/__init__.py") for p in resolved
                            ):
                                outgoing_reexport_resolved += 1
                            candidate_set |= set(resolved)
                        else:
                            outgoing_resolution_fallbacks += 1
                            # Per-fact fallback: keep direct import candidates so
                            # one successfully resolved import does not suppress
                            # other unresolved imports in the same file.
                            fact_fallback: set[str] = {
                                candidate_norm
                                for candidate_norm in (
                                    path_utils.normalize_repo_rel_path(candidate)
                                    for candidate in fact.candidate_paths
                                )
                                if (
                                    candidate_norm is not None
                                    and candidate_norm.endswith(".py")
                                    and candidate_norm != path
                                )
                            }
                            candidate_set |= fact_fallback

                # Keep broad fallback behavior when precise usage facts are absent.
                if (not usage_facts and candidates) or (usage_facts and not candidate_set and candidates):
                    candidate_set |= set(candidates)

                candidate_set, removed_init_count = _prune_intermediate_init_paths(
                    candidate_set,
                )
                outgoing_init_pruned += removed_init_count

                candidate_paths = sorted(candidate_set)
                if candidate_paths:
                    outgoing: dict[str, str] = {}
                    for candidate in candidate_paths:
                        # Priority 1: annotated patched content (dep changed in same snapshot)
                        dep_entry = path_map.get(candidate)
                        if dep_entry is not None:
                            patched = dep_entry.get("patched_content")
                            if patched is not None:
                                outgoing[candidate] = patched
                                continue
                        # Priority 2: raw HEAD content from snapshot zipball
                        dep_content = await _get_raw_dep_content(candidate)
                        if dep_content is not None:
                            outgoing[candidate] = dep_content
                    if outgoing:
                        file_entry["outgoing_dependencies"] = outgoing

                # ---- incoming dependencies ---------------------------------- #
                changed_symbols = file_entry.pop("_changed_symbols", None)
                if changed_symbols and usage_map:
                    # Normalised set of paths that the import resolver could emit
                    # for this file (direct, package/module counterpart, and
                    # source-root cross-variants).  Used to confirm that a
                    # candidate truly imports the changed module, not just
                    # mentions the same symbol name by coincidence.
                    target_variants = _incoming_import_target_variants(
                        path, snapshot_source_roots,
                    )
                    # One-hop re-export expansion: if any __init__.py in the
                    # changed file's package hierarchy re-exports from this
                    # module, callers that import through that __init__.py
                    # are also valid incoming dependents.
                    if reexport_map:
                        _extended: set[str] = set(target_variants)
                        for _variant in target_variants:
                            _extended |= reexport_map.get(_variant, frozenset())
                        target_variants = frozenset(_extended)
                    # Symbol regex is a cheap prefilter executed before the
                    # heavier AST import parse on each candidate.
                    file_pattern = re.compile(
                        r"\b(?:"
                        + "|".join(re.escape(s) for s in sorted(changed_symbols))
                        + r")\b",
                    )
                    incoming: dict[str, str] = {}
                    for dep_path, dep_content in usage_map.items():
                        if dep_path == path:
                            continue
                        if not dep_content or not file_pattern.search(dep_content):
                            continue
                        # Import-link validation: accept only when the candidate
                        # file has an import statement that resolves to the
                        # changed module (or one of its package-init equivalents
                        # under any recognised source root).
                        dep_imports, _ = import_resolution.resolve_import_candidates(
                            dep_content, dep_path, snapshot_source_roots,
                        )
                        if dep_imports & target_variants:
                            # Prefer annotated patched content when incoming dep was
                            # also changed in this snapshot (mirrors outgoing priority).
                            # Pattern matching/import validation use raw text so the
                            # AST parser is not confused by +/- line prefixes.
                            dep_patched_entry = path_map.get(dep_path)
                            if dep_patched_entry is not None:
                                dep_patched = dep_patched_entry.get("patched_content")
                                if dep_patched is not None:
                                    incoming[dep_path] = dep_patched
                                    continue
                            incoming[dep_path] = dep_content
                    if incoming:
                        file_entry["incoming_dependencies"] = incoming

            if (
                outgoing_imports_analyzed > 0
                or outgoing_reexport_resolved > 0
                or outgoing_resolution_fallbacks > 0
                or outgoing_init_pruned > 0
            ):
                logger.debug(
                    "Outgoing dependency resolution stats for %s @ %s: "
                    "imports_analyzed=%d, reexports_resolved=%d, "
                    "fallbacks=%d, init_pruned=%d",
                    repo_name,
                    snapshot_commit[:7],
                    outgoing_imports_analyzed,
                    outgoing_reexport_resolved,
                    outgoing_resolution_fallbacks,
                    outgoing_init_pruned,
                )

            # Drop transient parse helper content after dependency attachment.
            for _entry in path_map.values():
                if isinstance(_entry, dict):
                    _entry.pop("_head_text", None)

    # Phase 8: attach metadata files at the commit level under "metadata_files".
    # Content is an annotated diff when the
    # file changed between base and head; otherwise raw HEAD text.

    for pr_number in list(dataset[repo_name].keys()):
        pr_entry = dataset[repo_name][pr_number]
        base_commit = pr_entry.get("base_commit")
        base_meta = base_metadata_caches.get(base_commit, {}) if base_commit else {}
        for snapshot_commit, path_map in pr_entry["commits"].items():
            head_meta = snapshot_metadata_head.get(snapshot_commit, {})
            if not head_meta:
                continue
            cmap = compare_cache.get((pr_number, snapshot_commit), {})
            result: dict[str, str] = {}
            for meta_path, head_text in head_meta.items():
                cinfo = cmap.get(meta_path)
                if cinfo and cinfo.get("patch"):
                    base_text = base_meta.get(meta_path, "")
                    result[meta_path] = patches.full_file_annotated_diff(base_text, head_text)
                else:
                    result[meta_path] = head_text
            if result:
                path_map["metadata_files"] = result


async def enrich_dataset_with_code(
    dataset: MutableMapping[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    """Enrich the dataset with file content, patch annotations, dependencies, and metadata.

    For every repo this function runs :func:`_enrich_one_repo`, which performs
    all eight enrichment phases in a single pass:

    1. Fetch compare patches (one REST call per snapshot commit).
    2. Augment snapshots with balanced no-comment `.py` files.
    3. Fetch base-commit zipballs (one per unique `base_commit`); extract both
       Python file base content and metadata file base content via
       :func:`_scan_metadata_in_zip_bytes`.
    4. Apply patches inline — sets `patched_content` and annotated comment
       line numbers on each file entry.
    5. Parse Python imports from HEAD file text; collect all in-repo outgoing
       dependency candidates (including files also changed in the same snapshot).
       Also extract changed symbol names (functions, methods, classes) from each
       `.py` file for incoming-dependency search.
    6. Fetch snapshot-commit zipballs (one per unique `snapshot_commit`, covering
       all commits) to resolve unchanged in-repo dependency files, extract metadata
       HEAD content, and scan all `.py` files for word-boundary occurrences of
       the collected changed symbols.
    7. Attach `outgoing_dependencies: {path: content}` to each file entry for
       files it imports from.  Attach `incoming_dependencies: {path: content}`
       for `.py` files in the repo that both reference at least one of the
       changed symbols **and** contain an import statement that resolves to the
       changed file's module (or one of its package-init / source-root
       equivalents).  The import-link check eliminates cross-project false
       positives from common symbol names (e.g. `get_logger`, `create`).
    8. Attach `metadata_files: {path: content}` at the commit level.  Content
       is an annotated diff when the file changed between base and head;
       otherwise raw HEAD text.
    9. Attach `file_tree: {"tree": "..."}` at the commit level, where `tree`
       is a newline-separated sorted list of repository file paths at head.

    Repos are processed concurrently up to `config.GITHUB_API_CONCURRENCY`
    workers.

    Args:
        dataset:
            Nested dataset mapping (mutated in place).
        session:
            Shared aiohttp client session.
        semaphore:
            API concurrency limiter.
    """
    headers = config.github_api_headers()
    repo_names = list(dataset.keys())
    if not repo_names:
        dataset_utils.prune_empty_dataset(dataset)
        return

    worker_count = min(len(repo_names), config.GITHUB_API_CONCURRENCY)
    work_queue: asyncio.Queue[str | object] = asyncio.Queue()
    for name in repo_names:
        await work_queue.put(name)
    for _ in range(worker_count):
        await work_queue.put(_REPO_ENRICH_QUEUE_STOP)

    pbar = tqdm(total=len(repo_names), desc="Enriching repos")

    async def _repo_worker() -> None:
        while True:
            item = await work_queue.get()
            if item is _REPO_ENRICH_QUEUE_STOP:
                break
            name = item
            try:
                await _enrich_one_repo(dataset, name, session, semaphore, headers)
            except Exception as exc:
                logger.error("Repo enrichment failed for %s: %s", name, exc)
            finally:
                pbar.update(1)

    try:
        await asyncio.gather(*(_repo_worker() for _ in range(worker_count)))
    finally:
        pbar.close()
    dataset_utils.prune_empty_dataset(dataset)
