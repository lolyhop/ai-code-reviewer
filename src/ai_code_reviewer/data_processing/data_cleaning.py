import json
import logging
import os
import re
import typing as tp

import pandas as pd
from tqdm import tqdm

from ai_code_reviewer.data_processing.llm_client import LLMClient
from ai_code_reviewer.data_processing.prompts import (
    COMMENT_CLASSIFICATION_PROMPT,
    NO_COMMENT_VALIDATION_PROMPT,
    VALID_CATEGORIES,
)
from ai_code_reviewer.models.metadata import (
    build_pull_request_metadata,
    build_repository_metadata,
)
from ai_code_reviewer.models.retriever import Retriever
from ai_code_reviewer.models.schema import ReviewSample

logger = logging.getLogger(__name__)

DEP_TOP_K = 5
TOKEN_LIMIT = 30_000
DOCSTRING_MAX_LINES = 3
COMMENT_BLOCK_MAX_LINES = 3
BLANK_RUN_MAX_LINES = 2
CHECKPOINT_EVERY = 50

def _tree_line_depth(line: str) -> int:
    match = re.search(r"[├└]", line)
    return (match.start() // 4 + 1) if match else 0


def _tree_line_name(line: str) -> str:
    match = re.search(r"[├└]── (.+)", line)
    return match.group(1).strip() if match else line.strip()


def truncate_file_tree(tree: str, max_depth: int = 2) -> str:
    """Keep only the first *max_depth* levels, skipping children of dot-dirs."""
    if not tree or not isinstance(tree, str):
        return ""
    result: tp.List[str] = []
    skip_children_of_depth: tp.Optional[int] = None

    for line in tree.splitlines():
        depth = _tree_line_depth(line)

        if skip_children_of_depth is not None and depth > skip_children_of_depth:
            continue
        skip_children_of_depth = None

        if depth > max_depth:
            continue

        name = _tree_line_name(line)
        if name.startswith(".") and name.endswith("/"):
            skip_children_of_depth = depth

        result.append(line)

    return "\n".join(result)


_DEF_RE = re.compile(r"^([+ \-]?)([ \t]*)((?:async\s+)?def\s+.+?:\s*)$")

NumberedLine = tp.Tuple[tp.Optional[int], str]


def _detect_indent(line: str) -> str:
    raw = line.lstrip("+-")
    return " " * (len(raw) - len(raw.lstrip()))


def _find_opening_quote(line: str) -> tp.Optional[str]:
    """Return the triple-quote that opens a multi-line string on *line*, or None."""
    raw = line.lstrip("+-")
    for q in ('"""', "'''"):
        idx = raw.find(q)
        if idx == -1:
            continue
        after = raw[idx + 3 :]
        if q not in after:
            return q
    return None


def strip_long_docstrings(
    text: str,
    max_lines: int = DOCSTRING_MAX_LINES,
) -> tp.List[NumberedLine]:
    """Replace multi-line triple-quoted strings longer than *max_lines*.

    Detects ``\"\"\"``, ``'''`` anywhere on the line (assignments,
    dict values, standalone docstrings, etc.).

    Returns (original_1based_lineno | None, text) tuples.
    """
    if not text or not isinstance(text, str):
        return []

    lines = text.splitlines()
    n = len(lines)
    result: tp.List[NumberedLine] = []
    i = 0

    while i < n:
        quote = _find_opening_quote(lines[i])

        if quote is None:
            result.append((i + 1, lines[i]))
            i += 1
            continue

        ds_start = i
        i += 1
        while i < n:
            if quote in lines[i].lstrip("+-"):
                break
            i += 1
        ds_end = i
        span = ds_end - ds_start + 1

        if span > max_lines:
            result.append((ds_start + 1, lines[ds_start]))
            indent = _detect_indent(lines[min(ds_start + 1, n - 1)])
            if not indent:
                indent = _detect_indent(lines[ds_start]) + "    "
            result.append((None, f"{indent}[docstring hidden]"))
            if ds_end < n:
                result.append((ds_end + 1, lines[ds_end]))
            i = ds_end + 1
        else:
            for j in range(ds_start, min(ds_end + 1, n)):
                result.append((j + 1, lines[j]))
            i = ds_end + 1

    return result


def collapse_unchanged_functions(
    numbered: tp.List[NumberedLine],
) -> tp.List[NumberedLine]:
    """Hide bodies of unchanged functions from pre-numbered lines."""
    if not numbered:
        return []

    texts = [text for _, text in numbered]
    n = len(texts)

    func_spans: tp.List[tp.Tuple[int, int, int]] = []
    for i, text in enumerate(texts):
        m = _DEF_RE.match(text)
        if not m:
            continue
        indent_len = len(m.group(2))
        body_start = i + 1
        body_end = body_start
        while body_end < n:
            raw = texts[body_end]
            stripped = raw.lstrip("+-").rstrip()
            if not stripped:
                body_end += 1
                continue
            cur_content = raw.lstrip("+-")
            cur_indent = len(cur_content) - len(cur_content.lstrip())
            if cur_indent <= indent_len:
                break
            body_end += 1
        if body_end > body_start:
            func_spans.append((i, body_start, body_end))

    changed = {i for i, t in enumerate(texts) if t and t[0] in ("+", "-")}

    result: tp.List[NumberedLine] = []
    skip_until = -1

    for i, (lineno, text) in enumerate(numbered):
        if i < skip_until:
            continue

        found = False
        for def_idx, body_start, body_end in func_spans:
            if i == def_idx:
                span = set(range(def_idx, body_end))
                if not span & changed:
                    result.append((lineno, text))
                    indent = _detect_indent(texts[body_start] if body_start < n else "")
                    if not indent:
                        indent = "    "
                    result.append((None, f"{indent}[function body is hidden]"))
                    skip_until = body_end
                    found = True
                break

        if not found:
            result.append((lineno, text))

    return result


_COMMENT_RE = re.compile(r"^[+ \-]?\s*#")


def collapse_comment_blocks(
    numbered: tp.List[NumberedLine],
    max_lines: int = COMMENT_BLOCK_MAX_LINES,
) -> tp.List[NumberedLine]:
    """Collapse consecutive ``#`` comment lines exceeding *max_lines*."""
    if not numbered:
        return []

    result: tp.List[NumberedLine] = []
    buf: tp.List[NumberedLine] = []

    def _flush() -> None:
        if len(buf) <= max_lines:
            result.extend(buf)
        else:
            result.extend(buf[:max_lines])
            result.append(
                (
                    None,
                    f"{_detect_indent(buf[0][1])}# ... [{len(buf) - max_lines} comment lines hidden]",
                )
            )
        buf.clear()

    for entry in numbered:
        lineno, text = entry
        if _COMMENT_RE.match(text):
            buf.append(entry)
        else:
            if buf:
                _flush()
            result.append(entry)

    if buf:
        _flush()

    return result


def collapse_blank_lines(
    numbered: tp.List[NumberedLine],
    max_consecutive: int = BLANK_RUN_MAX_LINES,
) -> tp.List[NumberedLine]:
    """Collapse runs of blank lines exceeding *max_consecutive*."""
    if not numbered:
        return []

    result: tp.List[NumberedLine] = []
    blank_count = 0

    for lineno, text in numbered:
        if not text.strip() or text.strip() in ("+", "-"):
            blank_count += 1
            if blank_count <= max_consecutive:
                result.append((lineno, text))
        else:
            blank_count = 0
            result.append((lineno, text))

    return result


def preprocess_patched(patched: str) -> tp.List[NumberedLine]:
    """Full preprocessing pipeline for patched content."""
    numbered = strip_long_docstrings(patched)
    numbered = collapse_comment_blocks(numbered)
    numbered = collapse_blank_lines(numbered)
    return collapse_unchanged_functions(numbered)


def _clean_dep_text(text: str) -> str:
    """Strip docstrings, long comment blocks, and excessive blank lines from dependency text."""
    numbered = strip_long_docstrings(text)
    numbered = collapse_comment_blocks(numbered)
    numbered = collapse_blank_lines(numbered)
    return "\n".join(t for _, t in numbered)


def _format_dependencies(deps: tp.Dict[str, str]) -> str:
    if not deps:
        return "None"
    sections = []
    for path, content in deps.items():
        cleaned = _clean_dep_text(content.strip())
        sections.append(f"--- {path} ---\n{cleaned}")
    return "\n\n".join(sections)


def _format_patched_content(path: str, numbered_lines: tp.List[NumberedLine]) -> str:
    parts: tp.List[str] = []
    for lineno, text in numbered_lines:
        if lineno is not None:
            parts.append(f"{lineno:>4} | {text}")
        else:
            parts.append(f"     | {text}")
    return f"File: {path}\n" + "\n".join(parts)


def _safe_int(val: tp.Any) -> tp.Union[int, str]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return "?"


def _format_comments(comments: tp.List[tp.Dict[str, tp.Any]]) -> str:
    if not comments:
        return "None"
    parts: tp.List[str] = []
    for i, c in enumerate(comments):
        body = c.get("body", c.get("comment", ""))
        start = _safe_int(c.get("annotated_start_line", c.get("start_line")))
        end = _safe_int(c.get("annotated_end_line", c.get("end_line")))
        parts.append(f"[Comment {i}] Lines {start}-{end}:\n{body}")
    return "\n\n".join(parts)


def _parse_comments_column(raw: tp.Any) -> tp.List[tp.Dict[str, tp.Any]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw or raw == "[]":
            return []
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    try:
        items = list(raw)
        return [x for x in items if isinstance(x, dict)]
    except TypeError:
        return []


def _parse_bool(raw: str) -> bool:
    """Extract true/false from a free-form LLM response."""
    token = raw.strip().lower().rstrip(".").strip()
    if token.startswith("true"):
        return True
    if token.startswith("false"):
        return False
    logger.warning("Could not parse boolean from LLM response: %s", raw[:100])
    return False


def _parse_classifications(raw: str, expected_count: int) -> tp.List[str]:
    """Parse ``<index>: <category>`` lines into a list of categories."""
    results: tp.Dict[int, str] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        left, _, right = line.partition(":")
        try:
            idx = int(left.strip())
        except ValueError:
            continue
        category = right.strip().lower().replace(" ", "_")
        if category in VALID_CATEGORIES:
            results[idx] = category
        else:
            results[idx] = "other"

    return [results.get(i, "other") for i in range(expected_count)]


def _metadata_files_to_dict(raw: tp.Any) -> tp.Dict[str, str]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        out: tp.Dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict):
                path = item.get("path", item.get("file_path", f"meta_{len(out)}"))
                content = item.get("patched_content", item.get("content", ""))
                out[path] = content
        return out
    return {}


def _any_to_dict(raw: tp.Any) -> tp.Dict[str, str]:
    """Convert list-of-dicts, dict, or NaN into {path: content}."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        items = list(raw)
    except TypeError:
        return {}
    out: tp.Dict[str, str] = {}
    for item in items:
        if isinstance(item, dict):
            path = item.get("path", item.get("file_path", f"dep_{len(out)}"))
            content = item.get("patched_content", item.get("content", ""))
            out[path] = content
    return out


def _row_to_review_sample(row: pd.Series) -> ReviewSample:
    return ReviewSample(
        repo=row.get("repo", ""),
        pr_number=int(row.get("pr_number", 0)),
        pr_title=row.get("pr_title", ""),
        pr_body=row.get("pr_body", ""),
        repo_star_count=int(row.get("repo_star_count", 0)),
        commit_sha=row.get("commit_sha", ""),
        path=row.get("path", ""),
        patched_content=row.get("patched_content", ""),
        outgoing_dependencies=_any_to_dict(row.get("outgoing_dependencies")),
        incoming_dependencies=_any_to_dict(row.get("incoming_dependencies")),
        metadata_files=_metadata_files_to_dict(row.get("metadata_files", {})),
    )

class Checkpoint:
    """Lightweight JSON-based checkpoint for resumable row processing."""

    def __init__(
        self,
        path: tp.Optional[str] = None,
        every: int = CHECKPOINT_EVERY,
    ) -> None:
        self.path = path
        self.every = every
        self._data: tp.Dict[str, tp.List] = {}
        self._count = 0

    def load(self) -> int:
        """Load existing checkpoint. Returns number of already-processed rows."""
        if not self.path or not os.path.exists(self.path):
            return 0
        with open(self.path) as f:
            self._data = json.load(f)
        self._count = len(next(iter(self._data.values()), []))
        logger.info("Checkpoint loaded: %d rows from %s", self._count, self.path)
        return self._count

    def append(self, **kwargs: tp.Any) -> None:
        """Append one row of results. Auto-saves every *every* rows."""
        for key, value in kwargs.items():
            self._data.setdefault(key, []).append(
                self._make_serializable(value)
            )
        self._count += 1
        if self.path and self._count % self.every == 0:
            self._save()

    @staticmethod
    def _make_serializable(v: tp.Any) -> tp.Any:
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        if isinstance(v, (str, int, float, list)):
            return v
        return str(v)

    def finalize(self) -> None:
        """Remove the checkpoint file (call only after output is saved)."""
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
            logger.info("Checkpoint removed: %s", self.path)

    def save_final(self) -> None:
        """Flush remaining data to the checkpoint file."""
        if self.path:
            self._save()

    def get(self, key: str) -> tp.List:
        return self._data.get(key, [])

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, ensure_ascii=False)
        logger.info("Checkpoint: %d rows -> %s", self._count, self.path)


class DataCleaningPipeline:
    """Validate and classify code-review data via LLM-as-judge."""

    def __init__(
        self,
        llm_client: tp.Optional[LLMClient] = None,
        dep_top_k: int = DEP_TOP_K,
        file_tree_max_depth: int = 2,
        token_limit: int = TOKEN_LIMIT,
    ) -> None:
        self.llm = llm_client
        self.retriever = Retriever(retriever_type="heuristic", top_k=dep_top_k)
        self.file_tree_max_depth = file_tree_max_depth
        self.token_limit = token_limit

    def _build_context(self, row: pd.Series) -> tp.Dict[str, str]:
        sample = _row_to_review_sample(row)

        incoming = self.retriever.retrieve(
            [sample], retrieval_field="incoming_dependencies"
        )[0]
        outgoing = self.retriever.retrieve(
            [sample], retrieval_field="outgoing_dependencies"
        )[0]

        numbered = preprocess_patched(row.get("patched_content", ""))

        return {
            "repository_metadata": build_repository_metadata(sample),
            "pull_request_metadata": build_pull_request_metadata(sample),
            "file_tree": truncate_file_tree(
                row.get("file_tree", ""), self.file_tree_max_depth
            )
            or "Not available",
            "patched_content": _format_patched_content(row.get("path", ""), numbered),
            "outgoing_dependencies": _format_dependencies(outgoing),
            "incoming_dependencies": _format_dependencies(incoming),
        }

    def _build_validation_prompt(self, row: pd.Series) -> str:
        ctx = self._build_context(row)
        return NO_COMMENT_VALIDATION_PROMPT.format(**ctx)

    def _build_classification_prompt(self, row: pd.Series) -> str:
        ctx = self._build_context(row)
        comments = _parse_comments_column(row.get("comments", []))
        ctx["comments"] = _format_comments(comments)
        return COMMENT_CLASSIFICATION_PROMPT.format(**ctx)

    def _exceeds_limit(self, prompt: str) -> bool:
        return len(prompt) / 4 > self.token_limit

    def _skip_message(self) -> str:
        return f"Too long (limit = {self.token_limit} tokens)"

    def validate_no_comment(self, row: pd.Series) -> tp.Tuple[tp.Union[bool, str], str]:
        """Returns (parsed_result, raw_response)."""
        prompt = self._build_validation_prompt(row)
        if self._exceeds_limit(prompt):
            msg = self._skip_message()
            return msg, msg
        raw = self.llm.generate(prompt, max_tokens=16, temperature=0.3)
        return _parse_bool(raw), raw

    def classify_comments(
        self, row: pd.Series
    ) -> tp.Tuple[tp.Union[tp.List[str], str], str]:
        """Returns (parsed_result, raw_response)."""
        comments = _parse_comments_column(row.get("comments", []))
        prompt = self._build_classification_prompt(row)
        if self._exceeds_limit(prompt):
            msg = self._skip_message()
            return msg, msg
        raw = self.llm.generate(prompt, max_tokens=256, temperature=0.3)
        return _parse_classifications(raw, expected_count=len(comments)), raw

    def build_prompts(
        self,
        df: pd.DataFrame,
        max_rows: tp.Optional[int] = None,
        checkpoint_path: tp.Optional[str] = None,
        checkpoint_every: int = CHECKPOINT_EVERY,
    ) -> pd.DataFrame:
        """Build prompts for every row without calling the LLM.

        Pass *checkpoint_path* to enable resume-on-crash.
        """
        df = df.copy()
        if max_rows is not None:
            df = df.head(max_rows)

        ckpt = Checkpoint(checkpoint_path, checkpoint_every)
        start = ckpt.load()

        skipped = 0
        for i in tqdm(
            range(start, len(df)), desc="Building prompts", initial=start, total=len(df)
        ):
            row = df.iloc[i]
            comments = _parse_comments_column(row.get("comments", []))
            if comments:
                prompt = self._build_classification_prompt(row)
                ptype = "classification"
            else:
                prompt = self._build_validation_prompt(row)
                ptype = "validation"

            if self._exceeds_limit(prompt):
                prompt = self._skip_message()
                skipped += 1

            ckpt.append(prompt_type=ptype, prompt=prompt)

        if skipped:
            logger.info(
                "Skipped %d rows exceeding token limit (%d)", skipped, self.token_limit
            )

        ckpt.save_final()
        df["prompt_type"] = ckpt.get("prompt_type")
        df["prompt"] = ckpt.get("prompt")
        return df, ckpt

    def run(
        self,
        df: pd.DataFrame,
        max_rows: tp.Optional[int] = None,
        checkpoint_path: tp.Optional[str] = None,
        checkpoint_every: int = CHECKPOINT_EVERY,
    ) -> pd.DataFrame:
        """Run cleaning with LLM calls.

        Pass *checkpoint_path* to enable resume-on-crash.
        """
        df = df.copy()
        if max_rows is not None:
            df = df.head(max_rows)

        ckpt = Checkpoint(checkpoint_path, checkpoint_every)
        start = ckpt.load()

        for i in tqdm(
            range(start, len(df)), desc="Cleaning", initial=start, total=len(df)
        ):
            row = df.iloc[i]
            comments = _parse_comments_column(row.get("comments", []))
            if comments:
                parsed, raw = self.classify_comments(row)
                ckpt.append(has_blocking=None, categories=parsed, raw_response=raw)
            else:
                parsed, raw = self.validate_no_comment(row)
                ckpt.append(has_blocking=parsed, categories=None, raw_response=raw)

        ckpt.save_final()
        df["has_blocking_issues"] = [
            str(v) if v is not None else "" for v in ckpt.get("has_blocking")
        ]
        df["comment_categories"] = [
            str(v) if v is not None else "" for v in ckpt.get("categories")
        ]
        df["llm_raw_response"] = [
            str(v) if v is not None else "" for v in ckpt.get("raw_response")
        ]
        return df, ckpt


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Data-cleaning pipeline")
    parser.add_argument("input", help="Path to input parquet file")
    parser.add_argument(
        "-o", "--output", default="cleaned_dataset.parquet", help="Output path"
    )
    parser.add_argument("-n", "--max-rows", type=int, default=None)
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Build prompts and save without calling the LLM",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=TOKEN_LIMIT,
        help="Max estimated tokens per prompt",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint file path for resume"
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--api-key",
        help="Yandex Cloud API key",
    )
    parser.add_argument(
        "--folder-id", help="Yandex Cloud folder ID"
    )
    parser.add_argument("--model", default="qwen3.5-35b-a3b-fp8/latest")
    args = parser.parse_args()

    data = pd.read_parquet(args.input)
    logger.info("Loaded %d rows from %s", len(data), args.input)

    if args.prompts_only:
        pipeline = DataCleaningPipeline(llm_client=None, token_limit=args.token_limit)
        result, ckpt = pipeline.build_prompts(
            data,
            max_rows=args.max_rows,
            checkpoint_path=args.checkpoint,
            checkpoint_every=args.checkpoint_every,
        )
    else:
        client = LLMClient(
            api_key=args.api_key,
            folder_id=args.folder_id,
            model=args.model,
        )
        pipeline = DataCleaningPipeline(llm_client=client, token_limit=args.token_limit)
        result, ckpt = pipeline.run(
            data,
            max_rows=args.max_rows,
            checkpoint_path=args.checkpoint,
            checkpoint_every=args.checkpoint_every,
        )

    result.to_parquet(args.output, index=False)
    logger.info("Saved %d rows to %s", len(result), args.output)
    ckpt.finalize()
