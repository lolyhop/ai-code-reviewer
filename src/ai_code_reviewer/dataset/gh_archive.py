from __future__ import annotations

import asyncio
import gzip
import io
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, MutableMapping

import aiohttp
from tqdm import tqdm

from ai_code_reviewer.dataset import config
from ai_code_reviewer.dataset.http import async_http_get_bytes
from ai_code_reviewer.dataset.paths import normalize_repo_rel_path

logger = logging.getLogger(__name__)


def make_dataset() -> MutableMapping[str, Any]:
    """Return an empty nested dataset structure (repo → PR → commits → paths → comments)."""
    return defaultdict(
        lambda: defaultdict(
            lambda: {
                "base_commit": None,
                "commits": defaultdict(lambda: defaultdict(lambda: {"comments": []})),
            }
        )
    )


def is_likely_english(
    text: str,
    threshold: float | None = None,
) -> bool:
    """Check if text is likely English by checking ASCII character ratio.

    Args:
        text:
            The text to check.
        threshold:
            The threshold for the ASCII character ratio. Defaults to
            ``config.IS_LIKELY_ENGLISH_THRESHOLD``.

    Returns:
        True if the text is likely English, False otherwise.
    """
    if threshold is None:
        threshold = config.IS_LIKELY_ENGLISH_THRESHOLD
    if not text:
        return False

    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / len(text)
    return ratio >= threshold


def get_hunk_context_bounds(diff_hunk: str) -> tuple[int | None, int | None]:
    """Parses a legacy diff_hunk to find the proxy context window.

    Args:
        diff_hunk:
            The diff_hunk to parse.

    Returns:
        A tuple containing:
            - The context start line
            - The context end line
    """
    if not diff_hunk:
        return None, None

    lines = diff_hunk.strip().splitlines()
    if not lines:
        return None, None

    header_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", lines[0])
    if not header_match:
        return None, None
    context_start_line = int(header_match.group(1)) - 1
    current_line = context_start_line - 1

    for line in lines[1:]:
        if line.startswith(" ") or line.startswith("+"):
            current_line += 1
    context_end_line = current_line

    return context_start_line, context_end_line


def parse_gh_archive_hour_bytes(compressed_data: bytes) -> MutableMapping[str, Any]:
    """Decompress GH Archive ``.json.gz`` bytes and build a partial dataset tree.

    Runs synchronously; callers should offload with ``asyncio.to_thread`` from async code.

    Args:
        compressed_data:
            Raw gzip-compressed NDJSON body.

    Returns:
        Nested dataset mapping for this hour (same shape as ``make_dataset()``).
    """
    dataset = make_dataset()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed_data)) as gz:
            for line in gz:
                try:
                    event = json.loads(line)
                    if event.get("type") != "PullRequestReviewCommentEvent":
                        continue

                    payload = event.get("payload") or {}
                    comment = payload.get("comment") or {}
                    pull_request = payload.get("pull_request") or {}

                    path = comment.get("path")
                    body = comment.get("body")
                    diff_hunk = comment.get("diff_hunk")
                    commit_id = comment.get("commit_id")
                    original_commit_id = comment.get("original_commit_id")
                    line_no = comment.get("line")
                    original_line = comment.get("original_line")
                    gh_start = comment.get("start_line")
                    side = comment.get("side")
                    in_reply_to_id = comment.get("in_reply_to_id")

                    if not path or not path.endswith(".py"):
                        continue
                    if not body or not diff_hunk:
                        continue
                    if side == "LEFT":
                        continue
                    if not is_likely_english(body):
                        continue
                    if in_reply_to_id:
                        continue

                    pr_url = pull_request.get("url")
                    if not pr_url:
                        continue

                    parts = pr_url.rstrip("/").split("/")
                    repo_name = "/".join(parts[-4:-2])
                    pr_number = parts[-1]

                    base = pull_request.get("base") or {}
                    base_commit = base.get("sha")

                    snapshot_commit = original_commit_id or commit_id
                    if not snapshot_commit:
                        continue

                    pr_entry = dataset[repo_name][pr_number]
                    if pr_entry["base_commit"] is None and base_commit:
                        pr_entry["base_commit"] = base_commit

                    if gh_start is not None and line_no is not None:
                        head_start_line = min(gh_start, line_no)
                        head_end_line = max(gh_start, line_no)
                    elif line_no is not None:
                        head_start_line = line_no
                        head_end_line = line_no
                    elif original_line is not None:
                        head_start_line = original_line
                        head_end_line = original_line
                    else:
                        head_start_line, head_end_line = get_hunk_context_bounds(
                            diff_hunk
                        )

                    if head_start_line is None or head_end_line is None:
                        continue

                    path_norm = normalize_repo_rel_path(path)
                    if path_norm is None or not path_norm.endswith(".py"):
                        continue

                    comment_record: dict[str, Any] = {
                        "body": body,
                        "diff_hunk": diff_hunk,
                        "head_start_line": head_start_line,
                        "head_end_line": head_end_line,
                        "comment_id": comment.get("id"),
                    }
                    path_map = pr_entry["commits"][snapshot_commit]
                    try:
                        path_map[path_norm]["comments"].append(comment_record)
                    except Exception:
                        fe = path_map.get(path_norm)
                        if fe is not None and not fe.get("comments"):
                            del path_map[path_norm]
                        raise
                except Exception as e:
                    logger.warning("Skipping comment due to error: %s", e)
    except Exception as e:
        logger.error("Error parsing GH Archive hour payload: %s", e)
    return dataset


async def fetch_pr_comments_for_hour(
    session: aiohttp.ClientSession,
    date_hour: datetime,
    semaphore: asyncio.Semaphore,
) -> MutableMapping[str, Any]:
    """Fetches and filters GitHub archive Pull Request comments for a specific date and hour.

    Args:
        session:
            The aiohttp client session.
        date_hour:
            The datetime object representing the date and hour.
        semaphore:
            The semaphore for rate limiting.

    Returns:
        A dictionary containing the dataset.
    """
    url = f"{config.GH_ARCHIVE_API_BASE}/{date_hour.strftime('%Y-%m-%d-%-H')}.json.gz"
    dataset = make_dataset()

    try:
        status, compressed_data = await async_http_get_bytes(
            session, url, semaphore=semaphore
        )
        if status != 200:
            return dataset

        return await asyncio.to_thread(parse_gh_archive_hour_bytes, compressed_data)
    except Exception as e:
        logger.error("Error fetching %s: %s", url, e)
    return dataset


def merge_datasets(
    target: MutableMapping[str, Any],
    source: MutableMapping[str, Any],
) -> None:
    """Merge `source` into `target` in place.

    For each repo, PR, commit, and path, appends comments from `source`,
    skipping comments whose `comment_id` already exists in `target` for
    that same file and commit snapshot. Path keys are normalized with
    ``normalize_repo_rel_path`` so equivalent raw paths merge into one entry.

    `base_commit` is taken from `source` only when `target` has no
    `base_commit` yet (first merged hour wins).

    Args:
        target:
            Dataset to merge into (mutated).
        source:
            Hourly dataset to merge from.
    """
    for repo_name, pr_map in source.items():
        for pr_number, pr_entry in pr_map.items():
            tgt_pr = target[repo_name][pr_number]
            if tgt_pr["base_commit"] is None and pr_entry.get("base_commit"):
                tgt_pr["base_commit"] = pr_entry["base_commit"]
            for commit_commit, path_map in pr_entry["commits"].items():
                for path, file_entry in path_map.items():
                    path_norm = normalize_repo_rel_path(path)
                    if path_norm is None:
                        logger.warning(
                            "Skipping merge for unsafe path key %r in %s PR %s",
                            path,
                            repo_name,
                            pr_number,
                        )
                        continue
                    tgt_file = tgt_pr["commits"][commit_commit][path_norm]
                    existing_ids = {
                        c["comment_id"]
                        for c in tgt_file["comments"]
                        if c.get("comment_id") is not None
                    }
                    for comment in file_entry["comments"]:
                        cid = comment.get("comment_id")
                        if cid is not None and cid in existing_ids:
                            continue
                        tgt_file["comments"].append(comment)
                        if cid is not None:
                            existing_ids.add(cid)


async def fetch_pr_comments_range(
    session: aiohttp.ClientSession,
    start: datetime,
    end: datetime,
    semaphore: asyncio.Semaphore,
) -> MutableMapping[str, Any]:
    """Fetch GH Archive PR review comments for every hour from `start` through `end`.

    `start` and `end` are normalized to hour boundaries. All hours in the inclusive range
    `[start_hour, end_hour]` are fetched in parallel, then merged into one dataset.
    Duplicate comments (same `comment_id` on the same file/commit) are kept once.
    Hour processing uses at most ``config.GH_ARCHIVE_CONCURRENCY`` worker coroutines;
    concurrent HTTP remains limited by the passed semaphore (``GH_ARCHIVE_CONCURRENCY``).

    Args:
        session:
            The aiohttp client session.
        start:
            Range start (inclusive after normalization).
        end:
            Range end (inclusive after normalization).
        semaphore:
            The semaphore for rate limiting.

    Returns:
        A single merged dataset.

    Raises:
        ValueError:
            If the normalized start hour is after the normalized end hour.
    """
    start_h = start.replace(minute=0, second=0, microsecond=0)
    end_h = end.replace(minute=0, second=0, microsecond=0)
    if start_h > end_h:
        raise ValueError(
            f"start hour {start_h!r} is after end hour {end_h!r} after normalization"
        )
    hours: list[datetime] = []
    cur = start_h
    while cur <= end_h:
        hours.append(cur)
        cur += timedelta(hours=1)

    n = len(hours)
    if n == 0:
        return make_dataset()

    worker_count = min(n, config.GH_ARCHIVE_CONCURRENCY)
    results: list[MutableMapping[str, Any]] = [make_dataset() for _ in range(n)]
    work_queue: asyncio.Queue[tuple[int, datetime] | None] = asyncio.Queue()
    for i, hour in enumerate(hours):
        await work_queue.put((i, hour))
    for _ in range(worker_count):
        await work_queue.put(None)

    async def _hour_worker(pbar: tqdm) -> None:
        while True:
            item = await work_queue.get()
            if item is None:
                break
            idx, hour = item
            try:
                results[idx] = await fetch_pr_comments_for_hour(
                    session, hour, semaphore
                )
            except Exception as exc:
                logger.error("Hour fetch failed for %s: %s", hour, exc)
            finally:
                pbar.update(1)

    with tqdm(total=n, desc="Fetching hourly data") as pbar:
        await asyncio.gather(
            *(_hour_worker(pbar) for _ in range(worker_count)),
        )

    merged = make_dataset()
    for part in results:
        merge_datasets(merged, part)
    return merged


def filter_dataset_by_top_snapshot_commits(
    dataset: MutableMapping[str, Any],
    snapshot_commits_to_keep: int,
) -> MutableMapping[str, Any]:
    """Keep the top-N `(repo, PR, snapshot_commit)` triples globally by comment count.

    For every triple, counts all comments under that `snapshot_commit` (summed
    across paths), sorts all triples by that count descending, takes the first
    `snapshot_commits_to_keep` triples, and copies only those commits into a
    new dataset with the same shape as `make_dataset()`.

    When counts tie, ordering is by `repo_name`, then `pr_number`, then
    `snapshot_commit` (all ascending) for reproducibility.

    PRs with no kept commits are omitted, as are repos with no PRs left. If
    `snapshot_commits_to_keep` is 0, returns an empty `make_dataset()` tree.

    Paths with no comments are skipped (defensive; ingestion uses normalized keys).

    Args:
        dataset:
            Dataset to filter.
        snapshot_commits_to_keep:
            Number of `(repo, PR, snapshot_commit)` triples to retain globally
            (non-negative).

    Returns:
        A new dataset with `defaultdict` structure matching `make_dataset()`.

    Raises:
        ValueError:
            If `snapshot_commits_to_keep` is negative.
    """
    if snapshot_commits_to_keep < 0:
        raise ValueError(
            f"snapshot_commits_to_keep must be >= 0, got {snapshot_commits_to_keep}"
        )

    scored: list[tuple[str, str, str, int]] = []
    for repo_name, pr_map in dataset.items():
        for pr_number, pr_entry in pr_map.items():
            for commit, path_map in pr_entry["commits"].items():
                total = sum(len(fe["comments"]) for fe in path_map.values())
                scored.append((repo_name, pr_number, commit, total))

    scored.sort(key=lambda t: (-t[3], t[0], t[1], t[2]))
    keep = {(r, p, s) for r, p, s, _ in scored[:snapshot_commits_to_keep]}

    out = make_dataset()
    for repo_name, pr_map in dataset.items():
        for pr_number, pr_entry in pr_map.items():
            tgt_pr = None
            for commit, path_map in pr_entry["commits"].items():
                if (repo_name, pr_number, commit) not in keep:
                    continue
                if tgt_pr is None:
                    tgt_pr = out[repo_name][pr_number]
                    tgt_pr["base_commit"] = pr_entry.get("base_commit")
                for path, file_entry in path_map.items():
                    if len(file_entry.get("comments", [])) == 0:
                        continue
                    out_entry: dict[str, Any] = {
                        "comments": list(file_entry["comments"])
                    }
                    for key in ("base_content", "patch", "patched_content"):
                        if key in file_entry:
                            out_entry[key] = file_entry[key]
                    tgt_pr["commits"][commit][path] = out_entry
    return out


def prune_files_without_comments(dataset: MutableMapping[str, Any]) -> None:
    """Remove file entries whose ``comments`` list is empty (e.g. legacy checkpoints).

    Drops empty path maps, then empty commits, PRs, and repos. Mutates ``dataset`` in place.

    Args:
        dataset:
            Nested mapping from ``make_dataset()`` or JSON round-trip.
    """
    for repo_name in list(dataset.keys()):
        pr_map = dataset[repo_name]
        for pr_number in list(pr_map.keys()):
            commits = pr_map[pr_number]["commits"]
            for snap in list(commits.keys()):
                path_map = commits[snap]
                for path in list(path_map.keys()):
                    if len(path_map[path].get("comments", [])) == 0:
                        del path_map[path]
                if not commits[snap]:
                    del commits[snap]
            if not pr_map[pr_number]["commits"]:
                del pr_map[pr_number]
        if not dataset[repo_name]:
            del dataset[repo_name]
