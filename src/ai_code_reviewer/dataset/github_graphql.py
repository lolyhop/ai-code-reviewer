from __future__ import annotations

import asyncio
import logging
from collections.abc import MutableMapping
from typing import Any

import aiohttp
from tqdm import tqdm

import ai_code_reviewer.dataset.dataset_utils as dataset_utils
import ai_code_reviewer.dataset.http as http
from ai_code_reviewer.dataset import config


logger = logging.getLogger(__name__)

_GQL_STOP = object()

_REVIEW_THREADS_QUERY = """
query($owner: String!, $repo: String!, $pr: Int!, $after: String) {
  repository(owner: $owner, name: $repo) {
    stargazerCount
    pullRequest(number: $pr) {
      title
      body
      reviewThreads(first: 100, after: $after) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          isResolved
          comments(first: 100) {
            nodes {
              id
            }
          }
        }
      }
    }
  }
}
"""


async def fetch_review_threads_for_pr(
    owner: str,
    repo: str,
    pr_number_str: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str],
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Fetch review threads, PR metadata, and repository metadata for a single PR.

    PR-level fields (`title`, `body`) and repo-level fields
    (`stargazerCount`) are captured from the first successful page response
    and cached; subsequent pagination pages are used only for thread nodes.

    Args:
        owner:
            Repository owner login.
        repo:
            Repository name.
        pr_number_str:
            PR number as a string (parsed from the dataset).
        session:
            Shared aiohttp client session.
        semaphore:
            Concurrency limiter for API calls.
        headers:
            GraphQL request headers; must include `Authorization` when a
            token is available.

    Returns:
        ``(ok, threads, meta)`` — ``ok=False`` on HTTP/GraphQL error or missing PR
        (caller should drop); otherwise ``threads`` holds all review-thread nodes and
        ``meta`` holds ``{"title", "body", "star_count"}``.

    Raises:
        ValueError:
            If `pr_number_str` cannot be converted to an integer.
    """
    pr_int = int(pr_number_str)
    all_threads: list[dict[str, Any]] = []
    meta: dict[str, Any] | None = None
    cursor: str | None = None

    while True:
        variables: dict[str, Any] = {
            "owner": owner,
            "repo": repo,
            "pr": pr_int,
            "after": cursor,
        }
        status, data = await http.async_http_post_json(
            session,
            config.GITHUB_GRAPHQL_API,
            payload={"query": _REVIEW_THREADS_QUERY, "variables": variables},
            semaphore=semaphore,
            headers=headers,
        )

        if status != 200 or data is None:
            logger.warning(
                "GraphQL request failed for %s/%s PR %s: HTTP %s",
                owner,
                repo,
                pr_number_str,
                status,
            )
            return False, [], {}

        gql_errors = data.get("errors")
        if gql_errors:
            logger.warning(
                "GraphQL errors for %s/%s PR %s: %s",
                owner,
                repo,
                pr_number_str,
                gql_errors,
            )
            return False, [], {}

        repo_node: dict[str, Any] = (data.get("data") or {}).get("repository") or {}
        pull: dict[str, Any] | None = repo_node.get("pullRequest")
        if not pull:
            logger.warning(
                "PR %s not found in %s/%s via GraphQL; dropping PR.",
                pr_number_str,
                owner,
                repo,
            )
            return False, [], {}

        if meta is None:
            meta = {
                "title": pull.get("title"),
                "body": pull.get("body"),
                "star_count": repo_node.get("stargazerCount"),
            }

        review_threads = pull.get("reviewThreads") or {}
        nodes: list[dict[str, Any]] = review_threads.get("nodes") or []
        all_threads.extend(nodes)

        page_info = review_threads.get("pageInfo") or {}
        if page_info.get("hasNextPage"):
            cursor = page_info.get("endCursor")
        else:
            break

    return True, all_threads, meta or {}


def build_comment_resolution_index(
    threads: list[dict[str, Any]],
) -> dict[str, bool]:
    """Map each comment node ID to the `isResolved` flag of its parent thread.

    Args:
        threads:
            Raw `reviewThread` nodes as returned by
            `fetch_review_threads_for_pr`.

    Returns:
        Mapping of GraphQL comment node ID to `isResolved` boolean.
    """
    index: dict[str, bool] = {}
    for thread in threads:
        is_resolved: bool = bool(thread.get("isResolved", False))
        for comment in (thread.get("comments") or {}).get("nodes") or []:
            node_id: str | None = comment.get("id")
            if node_id:
                index[node_id] = is_resolved
    return index


def _populate_is_resolved_on_comments(
    dataset: MutableMapping[str, Any],
    resolution_index: dict[str, bool],
) -> None:
    """Set `is_resolved` on each comment and drop those absent from the index.

    Comments whose `comment_id` is not present in `resolution_index` are
    removed from their file entry because resolution status could not be
    determined (e.g. the comment was deleted or belongs to a different PR
    snapshot).

    Args:
        dataset:
            Dataset to enrich (mutated in place).
        resolution_index:
            Mapping of GraphQL comment node ID to `isResolved` bool, as
            produced by `build_comment_resolution_index`.
    """
    for pr_map in dataset.values():
        for pr_entry in pr_map.values():
            for path_map in pr_entry["commits"].values():
                for file_entry in path_map.values():
                    original = file_entry.get("comments", [])
                    kept = []
                    for c in original:
                        cid = c.get("comment_id")
                        if cid in resolution_index:
                            c["is_resolved"] = resolution_index[cid]
                            kept.append(c)
                    dropped = len(original) - len(kept)
                    if dropped:
                        logger.debug(
                            "Dropped %d comment(s) without is_resolved from a file entry.",
                            dropped,
                        )
                    file_entry["comments"] = kept


def _populate_pr_and_repo_metadata(
    dataset: MutableMapping[str, Any],
    meta_map: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Write PR-level and repository-level metadata into each PR entry.

    Fields written per PR entry:

    - `pr_title` — pull request title (`str | None`).
    - `pr_body` — pull request description body (`str | None`).
    - `repo_star_count` — repository star count at fetch time (`int | None`).

    Args:
        dataset:
            Dataset to enrich (mutated in place).
        meta_map:
            Mapping of `(repo_name, pr_number)` to a dict with keys
            `"title"`, `"body"`, and `"star_count"`.
    """
    for (repo_name, pr_number), meta in meta_map.items():
        pr_entry = (dataset.get(repo_name) or {}).get(pr_number)
        if pr_entry is not None:
            pr_entry["pr_title"] = meta.get("title")
            pr_entry["pr_body"] = meta.get("body")
            pr_entry["repo_star_count"] = meta.get("star_count")


async def enrich_dataset_with_graphql_info(
    dataset: MutableMapping[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> None:
    """Enrich each PR entry with PR metadata, repo metadata, and comment resolution.

    For every PR in the dataset this function:

    1. Fetches `title`, `body`, `stargazerCount`, and all
       `reviewThreads` (paginated) via a single GraphQL query per page.
    2. Stores `pr_title`, `pr_body`, and `repo_star_count` on the
       PR-level entry.
    3. Maps each inline comment to `is_resolved` via thread membership.
    4. Removes comments whose resolution status could not be determined and
       prunes empty repos / PRs that failed to fetch.

    Args:
        dataset:
            Dataset to enrich (mutated in place).
        session:
            Shared aiohttp client session.
        semaphore:
            API concurrency limiter (shared with other GitHub API calls).
    """
    headers = config.github_graphql_headers()

    pr_jobs: list[tuple[str, str]] = [
        (repo_name, pr_number)
        for repo_name, pr_map in dataset.items()
        for pr_number in pr_map
    ]

    if not pr_jobs:
        logger.info("No PRs in dataset; skipping GraphQL enrichment.")
        return

    resolution_index: dict[str, bool] = {}
    meta_map: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_repo_set: set[str] = set()
    failed_pr_set: set[tuple[str, str]] = set()

    worker_count = min(len(pr_jobs), config.GITHUB_API_CONCURRENCY)
    work_queue: asyncio.Queue[tuple[str, str] | object] = asyncio.Queue()
    for job in pr_jobs:
        await work_queue.put(job)
    for _ in range(worker_count):
        await work_queue.put(_GQL_STOP)

    pbar = tqdm(total=len(pr_jobs), desc="GraphQL enrichment")

    async def _worker() -> None:
        while True:
            item = await work_queue.get()
            if item is _GQL_STOP:
                break
            repo_name, pr_number_str = item
            parts = repo_name.split("/", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                logger.warning(
                    "Invalid repo_name %r; dropping repo from dataset.",
                    repo_name,
                )
                invalid_repo_set.add(repo_name)
                pbar.update(1)
                continue
            owner, repo = parts
            try:
                ok, threads, meta = await fetch_review_threads_for_pr(
                    owner,
                    repo,
                    pr_number_str,
                    session,
                    semaphore,
                    headers,
                )
                if ok:
                    index = build_comment_resolution_index(threads)
                    resolution_index.update(index)
                    meta_map[(repo_name, pr_number_str)] = meta
                else:
                    failed_pr_set.add((repo_name, pr_number_str))
            except Exception as exc:
                logger.error(
                    "GraphQL enrichment failed for %s PR %s: %s",
                    repo_name,
                    pr_number_str,
                    exc,
                )
                failed_pr_set.add((repo_name, pr_number_str))
            finally:
                pbar.update(1)

    try:
        await asyncio.gather(*(_worker() for _ in range(worker_count)))
    finally:
        pbar.close()

    for repo_name in invalid_repo_set:
        if repo_name in dataset:
            logger.warning("Dropping repo %r (invalid name).", repo_name)
            del dataset[repo_name]

    for repo_name, pr_number in failed_pr_set:
        if repo_name in dataset and pr_number in dataset[repo_name]:
            del dataset[repo_name][pr_number]

    _populate_pr_and_repo_metadata(dataset, meta_map)
    _populate_is_resolved_on_comments(dataset, resolution_index)
    dataset_utils.prune_empty_dataset(dataset)
