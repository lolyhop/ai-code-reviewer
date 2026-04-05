from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def path_content_dict_to_rows(
    d: Mapping[str, str] | dict[str, str] | None,
) -> list[dict[str, str]]:
    """Convert a path→text map to sorted list rows ``{path, patched_content}``.

    Used for ``outgoing_dependencies``, ``incoming_dependencies``, and
    ``metadata_files``. Empty or missing maps become ``[]`` so Parquet column
    types stay stable.

    Args:
        d:
            Mapping from repository-relative path to file text (patched, raw
            snapshot, or annotated diff depending on enrichment phase).

    Returns:
        List of dicts with keys ``path`` and ``patched_content``, sorted by
        ``path``.
    """
    if not d:
        return []
    return [{"path": p, "patched_content": d[p]} for p in sorted(d)]


def _normalize_patched_content(text: str) -> str:
    """Normalize newlines and strip whitespace for deduplication keys."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _as_path_content_dict(raw: Any) -> dict[str, str]:
    """Coerce enrichment path→text values to a plain string dict.

    Args:
        raw:
            Value from the nested dataset (typically ``dict[str, str]`` or
            ``None``).

    Returns:
        Empty dict when ``raw`` is not a mapping; otherwise keys and values
        coerced to strings (``None`` values become empty strings).
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) if v is not None else "" for k, v in raw.items()}
    return {}


def build_files_list_rows(
    dataset: MutableMapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Flatten ``dataset`` to export rows with per-PR deduplication.

    Row grain: one row per deduplicated file snapshot within each PR
    (``dedupe_key = (path, has_comments, normalized_patched_content)``).

    Args:
        dataset:
            Nested dataset after enrichment.

    Returns:
        Tuple of ``(rows, stats)`` where ``stats`` contains counts
        ``raw_rows_total``, ``rows_kept_total``, ``rows_dropped_duplicates``,
        ``rows_dropped_with_comments``, ``rows_dropped_without_comments``.
    """
    rows: list[dict[str, Any]] = []
    raw_rows_total = 0
    rows_kept_total = 0
    rows_dropped_duplicates = 0
    rows_dropped_with_comments = 0
    rows_dropped_without_comments = 0

    for repo_name, pr_map in dataset.items():
        seen_keys: set[tuple[str, bool, str]] = set()
        for pr_number, pr_entry in pr_map.items():
            pr_title = pr_entry.get("pr_title")
            pr_body = pr_entry.get("pr_body")
            repo_star_count = pr_entry.get("repo_star_count")
            for commit_sha, path_map in pr_entry["commits"].items():
                commit_metadata_files = path_content_dict_to_rows(
                    _as_path_content_dict(path_map.get("metadata_files"))
                )
                file_tree_obj = path_map.get("file_tree") or {}
                commit_file_tree = file_tree_obj.get("tree")
                for path, file_entry in path_map.items():
                    if path in {"metadata_files", "file_tree"}:
                        continue
                    if not isinstance(file_entry, dict):
                        continue
                    raw_rows_total += 1
                    patched_content = file_entry.get("patched_content", "")
                    if not isinstance(patched_content, str):
                        patched_content = str(patched_content)
                    normalized = _normalize_patched_content(patched_content)
                    comments = file_entry.get("comments", [])
                    if not isinstance(comments, list):
                        comments = []
                    has_comments = bool(comments)
                    dedupe_key = (path, has_comments, normalized)
                    if dedupe_key in seen_keys:
                        rows_dropped_duplicates += 1
                        if has_comments:
                            rows_dropped_with_comments += 1
                        else:
                            rows_dropped_without_comments += 1
                        continue
                    seen_keys.add(dedupe_key)

                    out_raw = _as_path_content_dict(
                        file_entry.get("outgoing_dependencies")
                    )
                    in_raw = _as_path_content_dict(
                        file_entry.get("incoming_dependencies")
                    )

                    row: dict[str, Any] = {
                        "repo": repo_name,
                        "pr_number": pr_number,
                        "pr_title": pr_title,
                        "pr_body": pr_body,
                        "repo_star_count": repo_star_count,
                        "commit_sha": commit_sha,
                        "path": path,
                        "patched_content": patched_content,
                        "outgoing_dependencies": path_content_dict_to_rows(out_raw),
                        "incoming_dependencies": path_content_dict_to_rows(in_raw),
                        "metadata_files": commit_metadata_files,
                        "file_tree": commit_file_tree,
                        "comments": [
                            {
                                "body": c.get("body"),
                                "is_resolved": c.get("is_resolved"),
                                "annotated_start_line": c.get(
                                    "annotated_start_line"
                                ),
                                "annotated_end_line": c.get("annotated_end_line"),
                            }
                            for c in comments
                            if isinstance(c, dict)
                        ],
                    }
                    rows.append(row)
                    rows_kept_total += 1

    stats = {
        "raw_rows_total": raw_rows_total,
        "rows_kept_total": rows_kept_total,
        "rows_dropped_duplicates": rows_dropped_duplicates,
        "rows_dropped_with_comments": rows_dropped_with_comments,
        "rows_dropped_without_comments": rows_dropped_without_comments,
    }
    return rows, stats


def save_files_list_parquet(
    dataset: MutableMapping[str, Any],
    path: Path | str,
) -> tuple[Path, dict[str, int]]:
    """Write the flattened dataset to Parquet (zstd, PyArrow).

    Args:
        dataset:
            Nested enrichment dataset.
        path:
            Destination ``.parquet`` path; parent directories are created.

    Returns:
        Tuple of resolved ``path`` and the same stats dict as
        :func:`build_files_list_rows`.
    """
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    rows, stats = build_files_list_rows(dataset)
    df = pd.DataFrame(rows)
    df.to_parquet(final_path, engine="pyarrow", compression="zstd", index=False)
    logger.info(
        "Wrote files_list Parquet (%d rows, %d raw before dedupe) -> %s",
        stats["rows_kept_total"],
        stats["raw_rows_total"],
        final_path,
    )
    return final_path, stats
