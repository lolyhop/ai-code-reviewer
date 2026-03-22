from __future__ import annotations

import asyncio
import io
import logging
import zipfile
from collections import Counter, defaultdict
from typing import Any, MutableMapping

import aiohttp
from tqdm import tqdm

from ai_code_reviewer.dataset import config
from ai_code_reviewer.dataset.http import async_http_get_bytes, async_http_get_json
from ai_code_reviewer.dataset.paths import normalize_repo_rel_path

logger = logging.getLogger(__name__)


def _infer_zip_root_prefix(zf: zipfile.ZipFile) -> str | None:
    """Infer the GitHub archive root folder (first path segment) from a zipball.

    Args:
        zf:
            Open zip file.

    Returns:
        Root prefix ending with ``/``, or None if the archive is empty.
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
        File text at ``base_commit``, or None if missing.
    """
    for key in (path, cinfo.get("previous_filename"), cinfo.get("filename")):
        if not key:
            continue
        nk = normalize_repo_rel_path(key)
        if nk is None:
            continue
        t = base_cache.get(nk)
        if t is not None:
            return t
    return None


def _paths_for_base_zip_lookup(path_norm: str, cinfo: dict[str, Any]) -> set[str]:
    """Normalized repo paths to load from the base zipball.

    Mirrors the lookup order in ``_resolve_base_text_from_cache`` (path, rename old, new).

    Args:
        path_norm:
            Normalized path from the dataset entry.
        cinfo:
            Compare file entry for this path.

    Returns:
        Repo-relative paths to request from the zipball at ``base_commit``.
    """
    out: set[str] = set()
    for key in (path_norm, cinfo.get("previous_filename"), cinfo.get("filename")):
        if not key:
            continue
        nk = normalize_repo_rel_path(key)
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
        Dictionary mapping each path to content of file.
    """
    out: dict[str, str | None] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            root = _infer_zip_root_prefix(zf)
            if not root:
                return {p: None for p in paths}
            for path in paths:
                member_path = normalize_repo_rel_path(path)
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
                if info.file_size > config.MAX_FILE_BYTES:
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


async def fetch_paths_from_zipball(
    owner: str,
    repo: str,
    base_commit: str,
    paths: set[str],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
) -> tuple[int, dict[str, str | None]]:
    """Download zip at ``base_commit`` and extract requested paths (one REST archive call).

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
        A tuple containing:
            - HTTP status code
            - Dictionary mapping each path to content of file.
    """
    url = f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/zipball/{base_commit}"
    status, data = await async_http_get_bytes(
        session, url, semaphore=semaphore, headers=headers
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
            - A dictionary mapping each relevant path (`filename` and `previous_filename` for renames) to
            `{"patch": str | None, "status": str | None, "filename": str | None, "previous_filename": str | None}`.
            - A boolean indicating whether the compare returned exactly 300 files (possible truncation per GitHub).
    """
    url = (
        f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/compare/"
        f"{base_commit}...{head_commit}"
    )
    status, data = await async_http_get_json(
        session, url, semaphore=semaphore, headers=headers
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
            nfn = normalize_repo_rel_path(fn)
            if nfn:
                path_to_info[nfn] = info
        if prev:
            nprev = normalize_repo_rel_path(prev)
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
    """Run ``fetch_compare_patches`` for each meta with a bounded worker pool.

    Uses at most ``config.GITHUB_API_CONCURRENCY`` concurrent coroutines
    (not one task per snapshot). HTTP concurrency is still limited by ``semaphore``.

    Args:
        session:
            Shared aiohttp session.
        owner:
            Repository owner login.
        repo:
            Repository name.
        compare_metas:
            Ordered ``(pr_number, base_commit, snapshot_commit)`` tuples.
        semaphore:
            API concurrency limiter.
        headers:
            GitHub REST headers.

    Returns:
        One list entry per ``compare_metas`` element, in order: either result
        ``(path_to_info, maybe_truncated)`` or ``BaseException`` if the fetch failed.
    """
    n = len(compare_metas)
    if n == 0:
        return []
    results: list[tuple[dict[str, dict[str, Any]], bool] | BaseException | None] = [
        None
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


def _prune_empty_dataset(dataset: MutableMapping[str, Any]) -> None:
    """Remove empty commit maps, PRs, and repos after enrichment."""
    for repo_name in list(dataset.keys()):
        pr_map = dataset[repo_name]
        for pr_number in list(pr_map.keys()):
            commits = pr_map[pr_number]["commits"]
            for commit in list(commits.keys()):
                if not commits[commit]:
                    del commits[commit]
            if not commits:
                del pr_map[pr_number]
        if not pr_map:
            del dataset[repo_name]


async def _enrich_one_repo(
    dataset: MutableMapping[str, Any],
    repo_name: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
) -> None:
    """Enrich a single top-level repo key (compare, zipball, patch fields)."""
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
        for snapshot_commit in pr_entry["commits"].keys():
            compare_metas.append((pr_number, base_commit, snapshot_commit))

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
                (pr_number, base_commit, snapshot_commit, cmap, trunc)
            )

    compare_cache: dict[tuple[Any, str], dict[str, dict[str, Any]]] = {}
    for pr_number, base_commit, snapshot_commit, cmap, _trunc in compare_results:
        compare_cache[(pr_number, snapshot_commit)] = cmap

    base_to_paths: defaultdict[str, set[str]] = defaultdict(set)
    for pr_number in list(dataset[repo_name].keys()):
        pr_entry = dataset[repo_name][pr_number]
        base_commit = pr_entry.get("base_commit")
        if not base_commit:
            continue
        for snapshot_commit, path_map in pr_entry["commits"].items():
            cmap = compare_cache.get((pr_number, snapshot_commit), {})
            for path in path_map.keys():
                path_norm = normalize_repo_rel_path(path)
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

    base_caches: dict[str, dict[str, str | None]] = {}

    async def fetch_one_base(
        base_commit: str,
    ) -> tuple[str, int, dict[str, str | None]]:
        paths_needed = base_to_paths[base_commit]
        if not paths_needed:
            return base_commit, 200, {}
        status, mapping = await fetch_paths_from_zipball(
            owner,
            repo,
            base_commit,
            paths_needed,
            session,
            semaphore,
            headers,
        )
        return base_commit, status, mapping

    zip_tasks = [fetch_one_base(bs) for bs in base_to_paths.keys()]
    zip_results: list[tuple[str, int, dict[str, str | None]]] = []
    if zip_tasks:
        zip_raw = await asyncio.gather(*zip_tasks, return_exceptions=True)
        for bs, res in zip(base_to_paths.keys(), zip_raw):
            if isinstance(res, BaseException):
                logger.error(
                    "Zipball task failed for %s @ %s: %s",
                    repo_name,
                    bs[:7],
                    res,
                )
                zip_results.append((bs, 599, {}))
            else:
                zip_results.append(res)

    for base_commit, status, path_map in zip_results:
        if status != 200:
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
            base_caches[base_commit] = path_map

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
                file_entry = pr_entry["commits"][snapshot_commit][path]
                path_norm = normalize_repo_rel_path(path)
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
                        base_cache, path_norm, cinfo
                    )
                    if base_text is None:
                        del pr_entry["commits"][snapshot_commit][path]
                        continue
                    file_entry["base_content"] = base_text
                    file_entry["patch"] = patch


async def enrich_dataset_with_base_and_patches(
    dataset: MutableMapping[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    """Attach `base_content` and `patch` to each file entry.

    Compare runs first (parallel per snapshot). Zipball paths are chosen only for files
    that have a non-empty patch and are not ``added`` (those use empty base text), so the
    repository archive is not downloaded when compare yields no usable patches.
    Failures in one repository are logged and do not stop enrichment of other repos.
    At most ``config.GITHUB_API_CONCURRENCY`` coroutines process repositories at a time;
    HTTP parallelism remains limited by ``semaphore``.

    Args:
        dataset:
            Dataset to enrich.
        session:
            The aiohttp client session.
        semaphore:
            API concurrency limiter.
    """
    headers = config.github_api_headers()
    repo_names = list(dataset.keys())
    if not repo_names:
        _prune_empty_dataset(dataset)
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
    _prune_empty_dataset(dataset)
