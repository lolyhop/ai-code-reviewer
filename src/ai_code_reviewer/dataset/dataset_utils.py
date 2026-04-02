from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, MutableMapping

from ai_code_reviewer.dataset import config
from ai_code_reviewer.dataset.paths import normalize_repo_rel_path

logger = logging.getLogger(__name__)


def make_dataset() -> MutableMapping[str, Any]:
    """Return an empty nested dataset structure (repo → PR → commits → paths → comments).

    Returns:
        A four-level `defaultdict` tree whose leaves are `{"comments": []}`
        dicts and whose PR level carries a `"base_commit"` key.
    """
    return defaultdict(
        lambda: defaultdict(
            lambda: {
                "base_commit": None,
                "commits": defaultdict(lambda: defaultdict(lambda: {"comments": []})),
            }
        )
    )


def merge_datasets(
    target: MutableMapping[str, Any],
    source: MutableMapping[str, Any],
) -> None:
    """Merge `source` into `target` in place.

    For each repo, PR, commit, and path, appends comments from `source`,
    skipping comments whose `comment_id` (GraphQL node ID) already exists in
    `target` for that same file and commit snapshot.  Path keys are normalised
    with `normalize_repo_rel_path` so equivalent raw paths merge into one entry.

    `base_commit` is taken from `source` only when `target` has no
    `base_commit` yet (first merged hour wins).

    Args:
        target:
            Dataset to merge into (mutated).
        source:
            Dataset to merge from.
    """
    for repo_name, pr_map in source.items():
        for pr_number, pr_entry in pr_map.items():
            tgt_pr = target[repo_name][pr_number]
            if tgt_pr["base_commit"] is None and pr_entry.get("base_commit"):
                tgt_pr["base_commit"] = pr_entry["base_commit"]
            for snapshot_commit, path_map in pr_entry["commits"].items():
                for path, file_entry in path_map.items():
                    if path == config.METADATA_FILES_COMMIT_KEY:
                        continue
                    path_norm = normalize_repo_rel_path(path)
                    if path_norm is None:
                        logger.warning(
                            "Skipping merge for unsafe path key %r in %s PR %s",
                            path,
                            repo_name,
                            pr_number,
                        )
                        continue
                    tgt_file = tgt_pr["commits"][snapshot_commit][path_norm]
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


def prune_empty_dataset(dataset: MutableMapping[str, Any]) -> None:
    """Remove empty commit maps, PRs, and repos after enrichment.

    Iterates the dataset in place and deletes any commit snapshot that has no
    file entries, any PR that has no remaining commits, and any repo that has no
    remaining PRs.

    Args:
        dataset:
            Dataset to prune (mutated in place).
    """
    for repo_name in list(dataset.keys()):
        pr_map = dataset[repo_name]
        for pr_number in list(pr_map.keys()):
            commits = pr_map[pr_number]["commits"]
            for commit in list(commits.keys()):
                real_keys = [
                    k
                    for k in commits[commit]
                    if k != config.METADATA_FILES_COMMIT_KEY
                ]
                if not real_keys:
                    del commits[commit]
            if not commits:
                del pr_map[pr_number]
        if not pr_map:
            del dataset[repo_name]