from __future__ import annotations

import ast
import logging
import re

logger = logging.getLogger(__name__)

# Matches the new-file hunk header in a unified diff, e.g. `@@ -3,7 +10,5 @@`.
# Group 1: new-file start line (1-based).
# Group 2: new-file line count (optional; absent when count is 1).
_HUNK_HEADER_RE: re.Pattern[str] = re.compile(
    r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@"
)


def extract_patch_head_line_numbers(patch: str) -> set[int]:
    """Parse a unified diff and return 1-based HEAD line numbers of changed lines.

    Only lines that appear on the new-file (`+`) side of the diff are collected.
    Context lines and removed lines (`-`) are ignored.  Hunk headers (`@@`) and
    file-header lines (`+++`) are also skipped.

    Args:
        patch:
            Unified diff text, e.g. the `patch` field from the GitHub compare API.

    Returns:
        Set of 1-based line numbers in the HEAD file that were added or modified.
        Empty set when `patch` is empty or contains no `+` lines.
    """
    changed: set[int] = set()
    current_line: int = 0

    for line in patch.splitlines():
        m = _HUNK_HEADER_RE.match(line)
        if m:
            current_line = int(m.group(1))
            continue

        if line.startswith("+++"):
            continue

        if line.startswith("\\"):
            # "\ No newline at end of file" and similar diff meta-markers.
            # These are not real file lines and must not advance the HEAD counter.
            continue

        if line.startswith("+"):
            changed.add(current_line)
            current_line += 1
        elif line.startswith("-"):
            pass  # removed line; does not advance HEAD counter
        else:
            # Context line
            current_line += 1

    return changed


def extract_changed_symbols(
    head_text: str,
    changed_line_numbers: set[int],
    min_name_length: int = 3,
) -> set[str]:
    """Return names of named definitions whose bodies overlap with changed lines.

    Walks the AST of `head_text` and collects the `name` attribute of every
    :class:`ast.FunctionDef`, :class:`ast.AsyncFunctionDef`, and
    :class:`ast.ClassDef` node whose source range `[lineno, end_lineno]`
    contains at least one line from `changed_line_numbers`.

    Pure module-level procedural code (statements not inside any named
    definition) is intentionally ignored: if no named node encompasses a given
    changed line the line is silently skipped.

    Args:
        head_text:
            Full Python source text of the HEAD version of the file.
        changed_line_numbers:
            1-based line numbers of added/modified lines, as returned by
            :func:`extract_patch_head_line_numbers`.
        min_name_length:
            Minimum number of characters a symbol name must have to be
            included in the result.  Short names (e.g. `do`, `ok`) match
            too broadly during repo-wide search and are excluded.

    Returns:
        Set of symbol names (strings) of named definitions overlapping the
        changed lines, filtered to names of at least `min_name_length`
        characters.  Returns an empty set when `head_text` cannot be parsed,
        `changed_line_numbers` is empty, or all changed lines fall in
        module-level code.
    """
    if not changed_line_numbers:
        return set()

    try:
        tree = ast.parse(head_text)
    except SyntaxError as exc:
        logger.debug("SyntaxError parsing HEAD text for symbol extraction: %s", exc)
        return set()

    symbols: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Emit the class name only when the class definition header line
            # itself was changed.  Using the full body range would add the
            # class name whenever any nested method changed, flooding the
            # symbol set with broad class names and producing many false
            # positive candidates in the zip scan.
            if node.lineno in changed_line_numbers and len(node.name) >= min_name_length:
                symbols.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_end: int = getattr(node, "end_lineno", node.lineno)
            if any(node.lineno <= ln <= node_end for ln in changed_line_numbers):
                if len(node.name) >= min_name_length:
                    symbols.add(node.name)

    return symbols
