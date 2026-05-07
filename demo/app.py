"""Streamlit demo: ``streamlit run demo/app.py``.

Loads the reviewer backend once (`streamlit.cache_resource`), then accepts a
GitHub PR URL, fetches data, runs :class:`~ai_code_reviewer.models.pipeline.ReviewPipeline`,
runs inference (transformers / ollama / openai per env), renders results.
"""

from __future__ import annotations

import os
import sys
import typing as tp
from pathlib import Path


_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import streamlit as st  # noqa: E402

from demo import view  # noqa: E402
from demo.adapters import (  # noqa: E402
    GitHubError,
    GitHubPRFetcher,
    PRReview,
    fetch_and_review_pr,
    get_review_model,
    parse_pr_url,
)


DEMO_PR_PLACEHOLDER = "https://github.com/owner/repo/pull/123"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_from_pr_review(review: PRReview) -> tp.Dict[str, tp.Any]:
    return review.as_dict()


def _python_file_count(files: tp.Sequence[tp.Mapping[str, tp.Any]]) -> int:
    return sum(1 for f in files if f.get("language") == "python")


def _blocking_issue_count(files: tp.Sequence[tp.Mapping[str, tp.Any]]) -> int:
    return sum(len(f.get("issues") or []) for f in files)


def _review_status(files: tp.Sequence[tp.Mapping[str, tp.Any]]) -> str:
    if any(f.get("status") == "blocking_issue" for f in files):
        return "changes_requested"
    if any(f.get("status") == "error" for f in files):
        return "partial"
    return "approved"


def _select_initial_file(files: tp.Sequence[tp.Mapping[str, tp.Any]]) -> int:
    for idx, f in enumerate(files):
        if f.get("status") == "blocking_issue":
            return idx
    for idx, f in enumerate(files):
        if f.get("status") == "clean":
            return idx
    return 0


# ---------------------------------------------------------------------------
# Model bootstrap
# ---------------------------------------------------------------------------


def _ensure_model_ready() -> tp.Optional[tp.Dict[str, tp.Any]]:
    """Warm backend; ``None`` if UI already showed an error."""
    cached_info = st.session_state.get("model_info")
    if cached_info and cached_info.get("loaded"):
        return cached_info

    backend = (
        os.environ.get("APR_BACKEND", "transformers").strip().lower() or "transformers"
    )
    if backend == "ollama":
        spinner_text = "Connecting to local Ollama server and warming up the model ..."
    elif backend == "openai":
        spinner_text = "Connecting to OpenAI-compatible API ..."
    else:
        spinner_text = "Loading local reviewer model (first run downloads weights) ..."

    with st.spinner(spinner_text):
        try:
            _model, info = get_review_model()
        except RuntimeError as exc:
            st.error(f"Local model error: {exc}")
            if backend == "ollama":
                st.info(
                    "Quick start for the Ollama backend:\n\n"
                    "```\n"
                    "brew install ollama\n"
                    "ollama serve &\n"
                    "ollama pull qwen2.5-coder:1.5b\n"
                    "```\n\n"
                    "Then reload this page.",
                )
            elif backend == "openai":
                st.info(
                    "```\n"
                    "export OPENAI_API_KEY=...\n"
                    "export OPENAI_BASE_URL=https://api.openai.com/v1\n"
                    "export APR_BACKEND=openai\n"
                    "```",
                )
            else:
                st.info(
                    '`pip install -e ".[finetune,demo]"` or `APR_BACKEND=ollama`.',
                )
            return None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while loading the model: {exc}")
            return None

    st.session_state["model_info"] = info
    return info


# ---------------------------------------------------------------------------
# Analysis flow
# ---------------------------------------------------------------------------


def _run_analysis(pr_url: str) -> tp.Optional[tp.Dict[str, tp.Any]]:
    progress = st.progress(0.0, text="Starting analysis ...")

    def _status(text: str, pct: float) -> None:
        progress.progress(min(max(pct, 0.0), 1.0), text=text)

    fetcher = GitHubPRFetcher()
    try:
        review = fetch_and_review_pr(pr_url, fetcher=fetcher, status=_status)
    except ValueError as exc:
        progress.empty()
        st.error(str(exc))
        return None
    except GitHubError as exc:
        progress.empty()
        st.error(f"GitHub API error: {exc}")
        if not os.environ.get("GITHUB_TOKEN"):
            st.warning(
                "No `GITHUB_TOKEN` was set. Public PRs may still be rate-limited; "
                "private PRs require a token. Export `GITHUB_TOKEN=...` and try again.",
            )
        return None
    except RuntimeError as exc:
        progress.empty()
        st.error(f"Local model error: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        st.error(f"Unexpected error during analysis: {exc}")
        return None

    progress.empty()
    if not review.files:
        st.warning("This PR contains no changed files.")
        return None
    if not any(f.language == "python" for f in review.files):
        st.warning(
            "This PR has no `.py` files. The reviewer only analyzes Python source files.",
        )
    st.success("Analysis complete.")
    return _result_from_pr_review(review)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="APR — Automated Pull Request Reviewer",
        page_icon="\U0001f50d",  # magnifying glass
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(view.CUSTOM_CSS, unsafe_allow_html=True)

    view.render_header()
    st.markdown("---")

    model_info = _ensure_model_ready()
    if model_info is None:
        return

    view.render_model_banner(model_info)

    if not os.environ.get("GITHUB_TOKEN"):
        st.warning(
            "`GITHUB_TOKEN` is not set. Public PRs may be rate-limited "
            "(60 requests/hour) and private PRs will fail. "
            "Set `GITHUB_TOKEN=ghp_...` in your environment for full access.",
        )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        pr_url = st.text_input(
            "GitHub Pull Request URL",
            value=st.session_state.get("pr_url", ""),
            placeholder=DEMO_PR_PLACEHOLDER,
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button(
            "Analyze PR",
            use_container_width=True,
            type="primary",
        )

    if analyze_clicked:
        st.session_state["pr_url"] = pr_url
        target_url = pr_url.strip()
        if not target_url:
            st.error("Please paste a GitHub PR URL.")
            result = None
        elif parse_pr_url(target_url) is None:
            st.error(
                "Invalid GitHub PR URL. Expected format: "
                "`https://github.com/owner/repo/pull/123`",
            )
            result = None
        else:
            result = _run_analysis(target_url)
        if result is not None:
            st.session_state["result"] = result
            st.session_state["selected_file_idx"] = _select_initial_file(
                result["files"]
            )

    result = st.session_state.get("result")
    if not result:
        return

    files = result["files"]
    meta = result["meta"]
    context = result["context"]

    st.markdown("### Review summary")
    view.render_summary_cards(
        meta=meta,
        n_python_files=_python_file_count(files),
        n_blocking_issues=_blocking_issue_count(files),
        review_status=_review_status(files),
    )

    sidebar_col, main_col = st.columns([1, 3])
    with sidebar_col:
        view.render_file_sidebar(
            files,
            selected_idx=st.session_state.get("selected_file_idx", 0),
        )
    with main_col:
        idx = st.session_state.get("selected_file_idx", 0)
        idx = max(0, min(idx, len(files) - 1))
        view.render_main_review_area(files[idx], file_idx=idx)

    st.markdown("---")
    view.render_context_section(context)


if __name__ == "__main__":
    main()
