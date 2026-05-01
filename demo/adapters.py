"""Thin integration layer between the Streamlit UI and the project pipeline.

Responsibilities (kept deliberately compact):

* Parse a GitHub PR URL.
* Fetch PR metadata, changed file list, and base file content via the GitHub
  REST API (synchronous ``urllib`` to play well with Streamlit).
* Reuse :func:`ai_code_reviewer.dataset.patches.compute_patched_content` to
  produce the same annotated full-file ``patched_content`` that the existing
  ``ReviewPipeline`` expects.
* Build :class:`ai_code_reviewer.models.schema.ReviewSample` rows and feed
  them into the existing :class:`ReviewPipeline` for prompt construction.
* Run inference with :class:`ai_code_reviewer.models.inference.ReviewModel`
  (the project's local Qwen3 baseline; configurable via env vars).
* Convert :class:`ReviewPrediction` into a UI-friendly structured dict.

This file does **not** rebuild the project's prompt logic — it only adapts
shapes and fetches live data. Whenever an equivalent module exists under
``src/ai_code_reviewer``, it is reused directly.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import typing as tp
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


from ai_code_reviewer.dataset.patches import compute_patched_content  # noqa: E402
from ai_code_reviewer.models.config import GenerationConfig, ModelConfig  # noqa: E402
from ai_code_reviewer.models.schema import (  # noqa: E402
    PredictedIssue,
    ReviewPrediction,
    ReviewSample,
)
from ai_code_reviewer.utils import parse_json_response  # noqa: E402


def _import_review_pipeline() -> tp.Any:
    """Lazy-import ``ReviewPipeline``.

    Done lazily because the import chain
    (``models.pipeline`` -> ``data_processing.data_cleaning`` ->
    ``data_processing.llm_client`` -> ``openai``) pulls in ``openai``, which
    is not in the project's base dependencies. Mock mode must keep working
    without it.
    """
    from ai_code_reviewer.models.pipeline import ReviewPipeline  # noqa: WPS433

    return ReviewPipeline


logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
USER_AGENT = "apr-demo/0.1 (+https://github.com/lolyhop/ai-code-reviewer)"

_GH_PR_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/pull/(?P<number>\d+)/?",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FileReview:
    path: str
    language: str
    status: str  # "blocking_issue" | "clean" | "skipped" | "error"
    additions: int
    deletions: int
    diff: str  # raw unified diff hunks (from GitHub API)
    source_lines: tp.List[tp.Tuple[int, str]]  # (annotated_lineno, "+/- /space" + text)
    issues: tp.List[tp.Dict[str, tp.Any]] = field(default_factory=list)
    error: tp.Optional[str] = None
    annotated_patched_content: str = ""

    def as_dict(self) -> tp.Dict[str, tp.Any]:
        return {
            "path": self.path,
            "language": self.language,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "diff": self.diff,
            "source_lines": self.source_lines,
            "issues": self.issues,
            "error": self.error,
        }


@dataclass
class PRReview:
    meta: tp.Dict[str, tp.Any]
    files: tp.List[FileReview]
    context: tp.Dict[str, tp.Any]
    is_mock: bool = False
    model_info: tp.Dict[str, tp.Any] = field(default_factory=dict)

    def as_dict(self) -> tp.Dict[str, tp.Any]:
        return {
            "meta": self.meta,
            "files": [f.as_dict() for f in self.files],
            "context": self.context,
            "is_mock": self.is_mock,
            "model_info": self.model_info,
        }


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def parse_pr_url(url: str) -> tp.Optional[tp.Dict[str, tp.Any]]:
    """Parse ``https://github.com/{owner}/{repo}/pull/{number}`` into parts."""
    m = _GH_PR_RE.match((url or "").strip())
    if not m:
        return None
    return {
        "owner": m.group("owner"),
        "repo": m.group("repo"),
        "number": int(m.group("number")),
    }


# ---------------------------------------------------------------------------
# GitHub fetcher (synchronous, stdlib-only)
# ---------------------------------------------------------------------------


class GitHubError(RuntimeError):
    """Raised when a GitHub API request fails."""


class GitHubPRFetcher:
    """Minimal synchronous GitHub client used by the demo only.

    The full project pipeline uses an async ``aiohttp``-based fetcher in
    ``ai_code_reviewer.dataset.github_api``. That implementation is designed
    for batch corpus enrichment and is too heavy to drive from a Streamlit
    callback, so the demo uses a small ``urllib`` adapter instead.
    """

    def __init__(self, token: tp.Optional[str] = None, timeout: float = 20.0) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.timeout = timeout

    def _request(self, url: str, *, accept: str = "application/vnd.github+json") -> bytes:
        req = urllib.request.Request(url)
        req.add_header("Accept", accept)
        req.add_header("User-Agent", USER_AGENT)
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise GitHubError(
                f"GitHub API HTTP {exc.code} for {url}\n{body}",
            ) from exc
        except urllib.error.URLError as exc:
            raise GitHubError(f"GitHub API network error for {url}: {exc.reason}") from exc

    def _get_json(self, url: str) -> tp.Any:
        return json.loads(self._request(url).decode("utf-8"))

    def fetch_pr(self, owner: str, repo: str, number: int) -> tp.Dict[str, tp.Any]:
        return self._get_json(f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}")

    def fetch_repo(self, owner: str, repo: str) -> tp.Dict[str, tp.Any]:
        return self._get_json(f"{GITHUB_API}/repos/{owner}/{repo}")

    def fetch_files(
        self,
        owner: str,
        repo: str,
        number: int,
        per_page: int = 100,
        max_pages: int = 5,
    ) -> tp.List[tp.Dict[str, tp.Any]]:
        out: tp.List[tp.Dict[str, tp.Any]] = []
        for page in range(1, max_pages + 1):
            url = (
                f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/files"
                f"?per_page={per_page}&page={page}"
            )
            chunk = self._get_json(url)
            if not isinstance(chunk, list) or not chunk:
                break
            out.extend(chunk)
            if len(chunk) < per_page:
                break
        return out

    def fetch_base_file(
        self,
        owner: str,
        repo: str,
        sha: str,
        path: str,
    ) -> tp.Optional[str]:
        """Return file content at ``sha``, or ``None`` for added/missing files."""
        try:
            url = f"{GITHUB_RAW}/{owner}/{repo}/{sha}/{urllib.parse.quote(path)}"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", USER_AGENT)
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            logger.warning("raw fetch failed %s: HTTP %s — falling back to contents API", path, exc.code)
        except urllib.error.URLError as exc:
            logger.warning("raw fetch failed %s: %s — falling back to contents API", path, exc.reason)
        contents_url = (
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/"
            f"{urllib.parse.quote(path)}?ref={sha}"
        )
        try:
            data = self._get_json(contents_url)
            if isinstance(data, dict) and data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except GitHubError as exc:
            logger.warning("contents fetch failed for %s @ %s: %s", path, sha, exc)
        return None


# ---------------------------------------------------------------------------
# Sample construction (reuses project schema and patch logic)
# ---------------------------------------------------------------------------


_PY_EXT = ".py"


def _is_python_path(path: str) -> bool:
    return path.endswith(_PY_EXT)


def _build_source_lines(annotated: str) -> tp.List[tp.Tuple[int, str]]:
    """Convert annotated full-file content into ``(lineno, prefix+text)`` tuples.

    Line numbers are 1-based positions inside the annotated string. They match
    the coordinate space used by ``_format_patched_content`` in the project
    pipeline, which is what the model sees in the prompt and what its returned
    line ranges refer to.
    """
    out: tp.List[tp.Tuple[int, str]] = []
    for idx, raw in enumerate(annotated.splitlines(), start=1):
        if not raw:
            out.append((idx, " "))
            continue
        first = raw[0]
        if first in {"+", "-"}:
            out.append((idx, raw))
        else:
            out.append((idx, " " + raw))
    return out


def _shrink_for_display(
    source_lines: tp.List[tp.Tuple[int, str]],
    max_lines: int = 200,
) -> tp.List[tp.Tuple[int, str]]:
    """Cap displayed lines so the diff viewer never blows up the page."""
    if len(source_lines) <= max_lines:
        return source_lines
    head = source_lines[: max_lines // 2]
    tail = source_lines[-max_lines // 2 :]
    truncated_marker = (
        head[-1][0] + 1,
        f"  ... {len(source_lines) - max_lines} lines hidden ...",
    )
    return head + [truncated_marker] + tail


def build_review_sample(
    *,
    owner: str,
    repo: str,
    pr: tp.Dict[str, tp.Any],
    file_entry: tp.Dict[str, tp.Any],
    base_content: tp.Optional[str],
    repo_star_count: int,
) -> tp.Tuple[tp.Dict[str, tp.Any], str, tp.List[tp.Tuple[int, str]]]:
    """Return ``(payload, annotated_text, source_lines)`` for one changed file.

    ``payload`` matches the dict shape consumed by ``ReviewSample.from_dict``.
    """
    path = file_entry["filename"]
    patch = file_entry.get("patch") or ""
    file_status = (file_entry.get("status") or "").lower()
    base_text = base_content if file_status != "added" and base_content is not None else ""

    annotated = ""
    source_lines: tp.List[tp.Tuple[int, str]] = []
    if patch:
        result = compute_patched_content(base_text, patch, path)
        if result is not None:
            annotated = result.annotated
            source_lines = _build_source_lines(annotated)

    payload: tp.Dict[str, tp.Any] = {
        "repo": f"{owner}/{repo}",
        "pr_number": int(pr.get("number", 0)),
        "pr_title": pr.get("title") or "",
        "pr_body": pr.get("body") or "",
        "repo_star_count": int(repo_star_count or 0),
        "commit_sha": (pr.get("head") or {}).get("sha") or "",
        "path": path,
        "patched_content": annotated,
        # The demo skips heavy dependency resolution — leave them empty.
        # ReviewPipeline handles empty dicts and prints "None" in the prompt.
        "outgoing_dependencies": {},
        "incoming_dependencies": {},
        "metadata_files": {},
        "file_tree": "",
        "comments": [],
    }
    return payload, annotated, source_lines


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _heuristic_issue_type(comment: str) -> str:
    text = (comment or "").lower()
    rules = [
        ("mutable default", "Blocking issue: default mutable argument"),
        ("race condition", "Blocking issue: race condition"),
        ("sql injection", "Blocking issue: SQL injection"),
        ("uncaught", "Blocking issue: uncaught exception"),
        ("unbounded", "Blocking issue: unbounded resource use"),
        ("none", "Blocking issue: None / null handling"),
        ("leak", "Blocking issue: resource leak"),
        ("auth", "Blocking issue: authorization"),
        ("typo", "Blocking issue: incorrect identifier"),
    ]
    for needle, label in rules:
        if needle in text:
            return label
    return "Blocking issue"


def prediction_to_issues(prediction: ReviewPrediction) -> tp.List[tp.Dict[str, tp.Any]]:
    """Convert a :class:`ReviewPrediction` to UI-friendly dicts."""
    issues: tp.List[tp.Dict[str, tp.Any]] = []
    for iss in prediction.issues:
        line_start = iss.line_start if iss.line_start is not None else 0
        line_end = iss.line_end if iss.line_end is not None else line_start
        comment = (iss.comment or "").strip()
        issues.append(
            {
                "type": _heuristic_issue_type(comment),
                "line_start": int(line_start),
                "line_end": int(line_end),
                "comment": comment,
                "suggestion": "",
                "confidence": 0.85,
            },
        )
    return issues


# ---------------------------------------------------------------------------
# Local model loader: dispatches between transformers and Ollama backends
# ---------------------------------------------------------------------------


def _model_config_from_env() -> ModelConfig:
    cfg = ModelConfig()
    cfg.model_name = os.environ.get("APR_MODEL_NAME", cfg.model_name)
    cfg.device_map = os.environ.get("APR_DEVICE", cfg.device_map)
    cfg.torch_dtype = os.environ.get("APR_TORCH_DTYPE", cfg.torch_dtype)
    cfg.load_in_4bit = os.environ.get("APR_LOAD_IN_4BIT", "").lower() in {"1", "true", "yes"}
    try:
        cfg.max_input_length = int(
            os.environ.get("APR_MAX_INPUT_TOKENS", cfg.max_input_length),
        )
    except ValueError:
        pass
    return cfg


def _gen_config_from_env() -> GenerationConfig:
    gen = GenerationConfig()
    try:
        gen.max_new_tokens = int(
            os.environ.get("APR_MAX_NEW_TOKENS", gen.max_new_tokens),
        )
    except ValueError:
        pass
    return gen


def _parse_review_response(raw: str) -> ReviewPrediction:
    """Parse model output (raw string) into a :class:`ReviewPrediction`.

    Mirrors :meth:`ReviewModel._parse_response` but lives here so the Ollama
    backend can use it without importing the torch-heavy ``ReviewModel``.
    """
    data = parse_json_response(raw)
    if not data:
        return ReviewPrediction()
    issues: tp.List[PredictedIssue] = []
    for item in data.get("issues", []):
        line_range = item.get("line_range") or {}
        issues.append(
            PredictedIssue(
                line_start=line_range.get("start"),
                line_end=line_range.get("end"),
                comment=item.get("comment", ""),
            ),
        )
    return ReviewPrediction(issues=issues)


# --- Ollama backend ---------------------------------------------------------


class _OllamaReviewModel:
    """Minimal Ollama client that mimics the ``ReviewModel.predict`` interface.

    Talks to Ollama's native HTTP API (``/api/chat`` with ``stream=False``).
    Recommended on Apple Silicon where ``transformers`` + MPS is much slower
    than llama.cpp via Metal (which Ollama uses under the hood).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 180.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _http_post_json(self, path: str, payload: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", USER_AGENT)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_get_json(self, path: str, timeout: tp.Optional[float] = None) -> tp.Dict[str, tp.Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", USER_AGENT)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health_check(self) -> tp.List[str]:
        """Verify the Ollama server is up and the requested model is pulled."""
        try:
            data = self._http_get_json("/api/tags", timeout=10.0)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama server not reachable at {self.base_url}.\n"
                "Start it with `ollama serve` (or `brew services start ollama`).",
            ) from exc
        models = [m.get("name", "") for m in (data.get("models") or [])]
        match = any(
            m == self.model_name or m.startswith(f"{self.model_name}:")
            for m in models
        )
        if not match:
            raise RuntimeError(
                f"Ollama model `{self.model_name}` is not pulled.\n"
                f"Run `ollama pull {self.model_name}` first.\n"
                f"Available models: {', '.join(models) if models else '(none)'}",
            )
        return models

    def warm_up(self) -> None:
        """Trigger Ollama to load the model into memory (avoids cold-start)."""
        try:
            self._http_post_json(
                "/api/chat",
                {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "options": {"num_predict": 1, "temperature": 0.0},
                },
            )
        except urllib.error.URLError as exc:
            logger.warning("Ollama warm-up failed (will still try to run): %s", exc)

    def generate_raw(
        self,
        prompt: str,
        gen_config: tp.Optional[GenerationConfig] = None,
    ) -> str:
        cfg = gen_config or GenerationConfig()
        options = {
            "num_predict": cfg.max_new_tokens,
            "temperature": cfg.temperature if cfg.do_sample else 0.0,
            "top_p": cfg.top_p,
            "repeat_penalty": cfg.repetition_penalty,
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Ollama supports forcing JSON-only output, which dramatically
            # improves parse success on small models.
            "format": "json",
            "options": options,
        }
        response = self._http_post_json("/api/chat", payload)
        return ((response.get("message") or {}).get("content")) or ""

    def predict(
        self,
        prompt: str,
        gen_config: tp.Optional[GenerationConfig] = None,
    ) -> ReviewPrediction:
        return _parse_review_response(self.generate_raw(prompt, gen_config))

    def predict_batch(
        self,
        prompts: tp.List[str],
        gen_config: tp.Optional[GenerationConfig] = None,
    ) -> tp.List[ReviewPrediction]:
        return [self.predict(p, gen_config) for p in prompts]


def _load_ollama_model() -> tp.Tuple[_OllamaReviewModel, tp.Dict[str, tp.Any]]:
    model_name = os.environ.get("APR_OLLAMA_MODEL", "qwen2.5-coder:1.5b")
    base_url = os.environ.get("APR_OLLAMA_URL", "http://localhost:11434")
    model = _OllamaReviewModel(model_name=model_name, base_url=base_url)
    model.health_check()
    model.warm_up()
    info = {
        "backend": "ollama",
        "name": f"ollama:{model_name}",
        "device": "Metal/CPU (via llama.cpp)",
        "endpoint": base_url,
        "max_input_tokens": None,
        "loaded": True,
    }
    return model, info


# --- Transformers backend (existing path) -----------------------------------


def _load_transformers_model() -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]:
    try:
        from ai_code_reviewer.models.inference import ReviewModel  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Local inference requires the `finetune` extras "
            "(torch, transformers). Install with:\n"
            "    pip install -e \".[finetune,demo]\"\n"
            "Or switch to the Ollama backend: APR_BACKEND=ollama.\n"
            f"Underlying import error: {exc}",
        ) from exc

    cfg = _model_config_from_env()
    model = ReviewModel(model_config=cfg)
    model.load()
    info = {
        "backend": "transformers",
        "name": cfg.model_name,
        "device": cfg.device_map,
        "torch_dtype": cfg.torch_dtype,
        "max_input_tokens": cfg.max_input_length,
        "loaded": True,
    }
    return model, info


# --- Dispatcher -------------------------------------------------------------


def _load_review_model() -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]:
    """Load the configured reviewer backend.

    Backend is chosen via the ``APR_BACKEND`` env var:

    * ``transformers`` (default) — local Qwen3 via HuggingFace transformers.
      Requires the ``finetune`` extras (torch, transformers).
    * ``ollama`` — local llama.cpp via Ollama. Recommended on Apple Silicon.
      Requires Ollama running locally (``ollama serve``) and the chosen
      model pulled (``ollama pull qwen2.5-coder:1.5b``).
    """
    backend = os.environ.get("APR_BACKEND", "transformers").strip().lower()
    if backend == "ollama":
        return _load_ollama_model()
    if backend in {"", "transformers", "hf", "huggingface"}:
        return _load_transformers_model()
    raise RuntimeError(
        f"Unknown APR_BACKEND={backend!r}. "
        "Supported values: 'transformers' (default) or 'ollama'.",
    )


def get_review_model() -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]:
    """Cached model loader.

    Wrapped with Streamlit's ``cache_resource`` if streamlit is available,
    otherwise behaves as a normal function.
    """
    try:
        import streamlit as st  # noqa: WPS433

        @st.cache_resource(show_spinner=False)
        def _cached() -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]:
            return _load_review_model()

        return _cached()
    except ImportError:
        return _load_review_model()


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


StatusCallback = tp.Callable[[str, float], None]


def _noop_status(_text: str, _progress: float) -> None:
    return None


def _build_context_summary(
    samples: tp.List[ReviewSample],
    pipeline_result: tp.Dict[str, tp.Any],
) -> tp.Dict[str, tp.Any]:
    """Summarize the prompt context for the "Context used by model" panel."""
    pr_title = ""
    pr_body = ""
    repo_summary = ""
    if samples:
        pr_title = samples[0].pr_title
        pr_body = samples[0].pr_body
    if pipeline_result.get("repo_metadata"):
        repo_summary = pipeline_result["repo_metadata"][0]

    changed_file_lines: tp.List[str] = []
    incoming_lines: tp.List[str] = []
    outgoing_lines: tp.List[str] = []
    incoming_per_file = pipeline_result.get("incoming") or []
    outgoing_per_file = pipeline_result.get("outgoing") or []
    for idx, sample in enumerate(samples):
        changed_file_lines.append(
            f"{sample.path}  (annotated {len(sample.patched_content.splitlines())} lines)",
        )
        if idx < len(incoming_per_file):
            for path in incoming_per_file[idx]:
                incoming_lines.append(f"{path} -> uses symbols changed in {sample.path}")
        if idx < len(outgoing_per_file):
            for path in outgoing_per_file[idx]:
                outgoing_lines.append(f"{sample.path} -> imports from {path}")

    return {
        "pr_title": pr_title,
        "pr_body": pr_body,
        "changed_file_context": "\n".join(changed_file_lines) or "Not available",
        "imported_definitions": outgoing_lines or ["(none retrieved for this PR)"],
        "usage_sites": incoming_lines or ["(none retrieved for this PR)"],
        "repo_metadata": repo_summary or "Not available",
    }


def fetch_and_review_pr(
    pr_url: str,
    *,
    fetcher: tp.Optional[GitHubPRFetcher] = None,
    status: StatusCallback = _noop_status,
    max_python_files: int = 6,
) -> PRReview:
    """End-to-end real PR review: fetch, build context, run inference, parse.

    All heavy lifting (prompt construction, retrieval, inference) is delegated
    to the existing project modules. This function only wires them together.
    """
    parsed = parse_pr_url(pr_url)
    if not parsed:
        raise ValueError(
            "Invalid GitHub PR URL. Expected format: "
            "https://github.com/{owner}/{repo}/pull/{number}",
        )

    fetcher = fetcher or GitHubPRFetcher()

    status("Fetching PR metadata", 0.05)
    pr = fetcher.fetch_pr(parsed["owner"], parsed["repo"], parsed["number"])
    repo_meta = {}
    try:
        repo_meta = fetcher.fetch_repo(parsed["owner"], parsed["repo"])
    except GitHubError as exc:
        logger.warning("repo metadata fetch failed: %s", exc)

    status("Loading changed Python files", 0.20)
    raw_files = fetcher.fetch_files(
        parsed["owner"], parsed["repo"], parsed["number"],
    )

    base_sha = (pr.get("base") or {}).get("sha") or ""
    head_sha = (pr.get("head") or {}).get("sha") or ""

    file_reviews: tp.List[FileReview] = []
    sample_payloads: tp.List[tp.Dict[str, tp.Any]] = []
    sample_annotated: tp.List[str] = []
    sample_source_lines: tp.List[tp.List[tp.Tuple[int, str]]] = []
    sample_owner_indices: tp.List[int] = []  # idx in file_reviews per sample

    py_count = 0
    for entry in raw_files:
        path = entry.get("filename") or ""
        additions = int(entry.get("additions") or 0)
        deletions = int(entry.get("deletions") or 0)
        diff = entry.get("patch") or ""

        if not _is_python_path(path):
            file_reviews.append(
                FileReview(
                    path=path,
                    language="other",
                    status="skipped",
                    additions=additions,
                    deletions=deletions,
                    diff="",
                    source_lines=[],
                    issues=[],
                    error=None,
                ),
            )
            continue

        if py_count >= max_python_files:
            file_reviews.append(
                FileReview(
                    path=path,
                    language="python",
                    status="skipped",
                    additions=additions,
                    deletions=deletions,
                    diff=diff,
                    source_lines=[],
                    issues=[],
                    error=f"Skipped: demo limit is {max_python_files} Python files per PR.",
                ),
            )
            continue
        py_count += 1

        try:
            base_content = fetcher.fetch_base_file(
                parsed["owner"], parsed["repo"], base_sha, path,
            )
            payload, annotated, source_lines = build_review_sample(
                owner=parsed["owner"],
                repo=parsed["repo"],
                pr=pr,
                file_entry=entry,
                base_content=base_content,
                repo_star_count=int(repo_meta.get("stargazers_count") or 0),
            )
        except Exception as exc:  # noqa: BLE001 — keep one bad file from killing demo
            logger.exception("failed to build sample for %s", path)
            file_reviews.append(
                FileReview(
                    path=path,
                    language="python",
                    status="error",
                    additions=additions,
                    deletions=deletions,
                    diff=diff,
                    source_lines=[],
                    issues=[],
                    error=f"Failed to build context: {exc}",
                ),
            )
            continue

        if not payload["patched_content"]:
            file_reviews.append(
                FileReview(
                    path=path,
                    language="python",
                    status="error",
                    additions=additions,
                    deletions=deletions,
                    diff=diff,
                    source_lines=[],
                    issues=[],
                    error="Could not apply patch (file may be binary or have an unusual diff).",
                ),
            )
            continue

        review = FileReview(
            path=path,
            language="python",
            status="clean",
            additions=additions,
            deletions=deletions,
            diff=diff,
            source_lines=_shrink_for_display(source_lines),
            issues=[],
            error=None,
            annotated_patched_content=annotated,
        )
        sample_owner_indices.append(len(file_reviews))
        file_reviews.append(review)
        sample_payloads.append(payload)
        sample_annotated.append(annotated)
        sample_source_lines.append(source_lines)

    status("Building repository context", 0.40)
    pipeline_result: tp.Dict[str, tp.Any] = {}
    samples: tp.List[ReviewSample] = []
    if sample_payloads:
        pipeline_cls = _import_review_pipeline()
        pipeline = pipeline_cls(retriever_type="heuristic", top_k=3)
        pipeline_result = pipeline.run(sample_payloads)
        samples = pipeline_result["samples"]

    model_info: tp.Dict[str, tp.Any] = {"loaded": False, "name": "(not loaded)"}
    predictions: tp.List[ReviewPrediction] = []
    if samples:
        status("Loading local model", 0.55)
        model, model_info = get_review_model()
        gen_config = _gen_config_from_env()
        prompts: tp.List[str] = pipeline_result["prompts"]
        status("Running inference", 0.75)
        for i, prompt in enumerate(prompts):
            try:
                predictions.append(model.predict(prompt, gen_config=gen_config))
            except Exception as exc:  # noqa: BLE001
                logger.exception("inference failed for sample %d", i)
                predictions.append(ReviewPrediction())
                target_idx = sample_owner_indices[i]
                file_reviews[target_idx].status = "error"
                file_reviews[target_idx].error = f"Inference failed: {exc}"

    status("Generating review comments", 0.95)
    for sample_idx, prediction in enumerate(predictions):
        target_idx = sample_owner_indices[sample_idx]
        review = file_reviews[target_idx]
        if review.status == "error":
            continue
        issues = prediction_to_issues(prediction)
        review.issues = issues
        review.status = "blocking_issue" if issues else "clean"

    context_summary = _build_context_summary(samples, pipeline_result)
    meta = {
        "repo": f"{parsed['owner']}/{parsed['repo']}",
        "pr_number": parsed["number"],
        "pr_title": pr.get("title") or "",
        "pr_body": pr.get("body") or "",
        "html_url": pr.get("html_url") or pr_url,
        "base_sha": base_sha[:7],
        "head_sha": head_sha[:7],
        "repo_star_count": int(repo_meta.get("stargazers_count") or 0),
    }

    status("Done", 1.0)
    return PRReview(
        meta=meta,
        files=file_reviews,
        context=context_summary,
        is_mock=False,
        model_info=model_info,
    )


# ---------------------------------------------------------------------------
# Public surface used by demo/app.py
# ---------------------------------------------------------------------------


__all__ = [
    "FileReview",
    "GitHubError",
    "GitHubPRFetcher",
    "PRReview",
    "build_review_sample",
    "fetch_and_review_pr",
    "get_review_model",
    "parse_pr_url",
    "prediction_to_issues",
]


# Re-exports for convenience in tests / notebooks.
PredictedIssue = PredictedIssue  # noqa: PLW0127
ReviewPrediction = ReviewPrediction  # noqa: PLW0127
ReviewSample = ReviewSample  # noqa: PLW0127
