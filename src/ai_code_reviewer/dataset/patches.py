from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, MutableMapping

from whatthepatch import apply_diff, parse_patch
from whatthepatch.exceptions import HunkApplyException

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PatchedContentResult:
    """Result of :func:`compute_patched_content`.

    Attributes:
        annotated: Full-file annotated diff view with `-`/`+` line prefixes.
        head_text: Clean HEAD file content after applying the patch (no markers).
        head_to_annotated: Mapping from 0-based HEAD line index to 0-based
            annotated view line index.
    """

    annotated: str
    head_text: str
    head_to_annotated: list[int]


def full_file_annotated_diff(base_text: str, head_text: str) -> str:
    """Interleave base and head so unchanged lines match base; mark edits with - / +.

    Args:
        base_text:
            Original file contents.
        head_text:
            Contents after applying the patch (same line endings as produced by the
            patch applier).

    Returns:
        Annotated diff string with unchanged lines unmarked and edits prefixed with ``-``/``+``.
    """
    a = base_text.splitlines(keepends=True)
    b = head_text.splitlines(keepends=True)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            out.extend(a[i1:i2])
        elif tag == "delete":
            out.extend(f"-{line}" for line in a[i1:i2])
        elif tag == "insert":
            out.extend(f"+{line}" for line in b[j1:j2])
        else:
            out.extend(f"-{line}" for line in a[i1:i2])
            out.extend(f"+{line}" for line in b[j1:j2])
    return "".join(out)


def map_head_lines_to_annotated_indices(
    base_lines: list[str],
    head_lines: list[str],
) -> list[int]:
    """Map each HEAD file line index (0-based) to annotated output line index (0-based).

    Args:
        base_lines:
            Base file lines (with `keepends`), aligned with `head_lines` for diffing.
        head_lines:
            HEAD file lines after applying the patch.

    Returns:
        For each head line `j`, `result[j]` is the 0-based line index in the annotated
        string that displays that head line.
    """
    if not head_lines:
        return []
    head_to_annotated = [0] * len(head_lines)
    annotated_idx = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, base_lines, head_lines
    ).get_opcodes():
        if tag == "equal":
            for j in range(j1, j2):
                head_to_annotated[j] = annotated_idx
                annotated_idx += 1
        elif tag == "delete":
            annotated_idx += i2 - i1
        elif tag == "insert":
            for j in range(j1, j2):
                head_to_annotated[j] = annotated_idx
                annotated_idx += 1
        else:
            annotated_idx += i2 - i1
            for j in range(j1, j2):
                head_to_annotated[j] = annotated_idx
                annotated_idx += 1
    return head_to_annotated


def head_blob_range_to_annotated_1based(
    head_blob_start: int | None,
    head_blob_end: int | None,
    head_to_annotated: list[int],
) -> tuple[int | None, int | None]:
    """Convert 1-based GitHub HEAD blob line numbers to 1-based annotated view line numbers.

    Args:
        head_blob_start:
            First line in the PR head file (1-based), or `None`.
        head_blob_end:
            Last line in the PR head file (1-based), or `None`.
        head_to_annotated:
            Mapping from `map_head_lines_to_annotated_indices` for the same base/head pair.

    Returns:
        `(annotated_start_line, annotated_end_line)` in 1-based form for `patched_content`,
        or `(None, None)` if out of range or inputs are invalid.
    """
    if head_blob_start is None or head_blob_end is None or not head_to_annotated:
        return None, None
    n = len(head_to_annotated)
    if not (1 <= head_blob_start <= n and 1 <= head_blob_end <= n):
        return None, None
    a0 = head_to_annotated[head_blob_start - 1]
    b0 = head_to_annotated[head_blob_end - 1]
    return a0 + 1, b0 + 1


def _with_trailing_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def apply_unified_patch(base_content: str, patch: str) -> str:
    """Apply a unified diff to `base_content` and return the resulting text.

    Args:
        base_content:
            File at the old side of the diff (empty for newly added files).
        patch:
            Unified diff text (e.g. GitHub compare `patch` field).

    Returns:
        Full file text after the patch, with a trailing newline when non-empty.

    Raises:
        ValueError:
            If `patch` does not parse to at least one diff.
        HunkApplyException:
            If the patch does not apply cleanly to `base_content`.
    """
    diffs = list(parse_patch(patch))
    if not diffs:
        raise ValueError("patch did not parse to any diff hunks")
    if len(diffs) > 1:
        logger.warning(
            "patch contained %d diff segments; applying only the first",
            len(diffs),
        )
    diff = diffs[0]
    lines = apply_diff(diff, base_content)
    if not lines:
        return ""
    return _with_trailing_newline("\n".join(lines))


def compute_patched_content(
    base_content: str,
    patch: str,
    path: str,
) -> PatchedContentResult | None:
    """Return annotated full-file view, clean HEAD text, and HEAD-to-annotated mapping.

    Args:
        base_content:
            Snapshot of the file before the change.
        patch:
            Unified diff for this file (e.g. from GitHub compare API).
        path:
            Repo-relative path (for logs only).

    Returns:
        :class:`PatchedContentResult` on success, or `None` if the patch could
        not be applied cleanly.
    """
    try:
        head = apply_unified_patch(base_content, patch)
    except (HunkApplyException, ValueError) as exc:
        logger.warning("could not apply patch for %s: %s", path, exc)
        return None
    base_norm = _with_trailing_newline(base_content)
    head_norm = _with_trailing_newline(head)
    a = base_norm.splitlines(keepends=True)
    b = head_norm.splitlines(keepends=True)
    head_to_annotated = map_head_lines_to_annotated_indices(a, b)
    annotated = full_file_annotated_diff(base_norm, head_norm)
    return PatchedContentResult(
        annotated=annotated,
        head_text=head,
        head_to_annotated=head_to_annotated,
    )


def enrich_dataset_with_patched_content(dataset: MutableMapping[str, Any]) -> None:
    """Set ``patched_content`` and per-comment ``annotated_*`` line fields.

    Standalone utility for applying patches outside the main enrichment pipeline.
    The canonical pipeline uses :func:`github_api.enrich_dataset_with_code`,
    which performs this step inline as phase 4.

    Args:
        dataset:
            Nested mapping produced by the EDA pipeline after base/patch enrichment.
    """
    for _, pr_map in dataset.items():
        for _pr_number, pr_entry in pr_map.items():
            commits = pr_entry.get("commits")
            if not commits:
                continue
            for _snapshot, path_map in commits.items():
                for path, file_entry in path_map.items():
                    if "base_content" not in file_entry or "patch" not in file_entry:
                        continue
                    base = file_entry["base_content"]
                    patch = file_entry["patch"]
                    if patch is None:
                        continue
                    base_str = base if isinstance(base, str) else ""
                    patch_str = patch if isinstance(patch, str) else ""
                    result = compute_patched_content(base_str, patch_str, path)
                    if result is None:
                        for comment in file_entry["comments"]:
                            comment["annotated_start_line"] = None
                            comment["annotated_end_line"] = None
                        continue
                    file_entry["patched_content"] = result.annotated
                    for comment in file_entry["comments"]:
                        a_s, a_e = head_blob_range_to_annotated_1based(
                            comment.get("head_start_line"),
                            comment.get("head_end_line"),
                            result.head_to_annotated,
                        )
                        comment["annotated_start_line"] = a_s
                        comment["annotated_end_line"] = a_e
