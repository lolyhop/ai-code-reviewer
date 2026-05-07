"""Streamlit layout and CSS for the demo."""

from __future__ import annotations

import html
import typing as tp

import streamlit as st


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Apply the body font only at the top level. Streamlit's internal text
   inherits naturally; targeting [class*="st-"] catches Streamlit's
   emotion-cache classes and may break expander/accessibility layout in
   newer versions. */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.apr-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
}
.apr-badge-blue   { background: #dbeafe; color: #1e40af; }
.apr-badge-green  { background: #d1fae5; color: #065f46; }
.apr-badge-purple { background: #ede9fe; color: #5b21b6; }
.apr-badge-amber  { background: #fef3c7; color: #92400e; }
.apr-badge-red    { background: #fee2e2; color: #991b1b; }
.apr-badge-gray   { background: #f1f5f9; color: #475569; }

.apr-badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 6px;
    align-items: center;
}
.apr-badge-row .apr-badge {
    margin-right: 0;
    margin-bottom: 0;
}

.apr-status-approved {
    display: inline-block; padding: 6px 16px; border-radius: 6px;
    background: #22c55e; color: white; font-weight: 700; font-size: 0.85rem;
}
.apr-status-changes-requested {
    display: inline-block; padding: 6px 16px; border-radius: 6px;
    background: #ef4444; color: white; font-weight: 700; font-size: 0.85rem;
}
.apr-status-error {
    display: inline-block; padding: 6px 16px; border-radius: 6px;
    background: #f59e0b; color: white; font-weight: 700; font-size: 0.85rem;
}

.apr-summary-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 16px; text-align: center; min-height: 90px;
}
.apr-summary-card .apr-summary-label {
    font-size: 0.75rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 6px;
}
.apr-summary-card .apr-summary-value {
    font-size: 1.3rem; font-weight: 700; color: #0f172a;
}

.apr-file-item {
    padding: 8px 12px; border-radius: 6px; margin-bottom: 4px;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 0.78rem; display: flex; align-items: center; gap: 8px;
    word-break: break-all;
}
.apr-file-item-error    { background: #fef2f2; border-left: 3px solid #ef4444; }
.apr-file-item-clean    { background: #f0fdf4; border-left: 3px solid #22c55e; }
.apr-file-item-skipped  { background: #f8fafc; border-left: 3px solid #94a3b8; color: #64748b; }
.apr-file-item-runtime  { background: #fffbeb; border-left: 3px solid #f59e0b; }

.apr-diff-block {
    background: #1e1e2e; color: #cdd6f4; border-radius: 8px;
    padding: 16px; font-family: 'SFMono-Regular', Consolas, monospace;
    font-size: 0.78rem; line-height: 1.55; overflow-x: auto;
    margin-bottom: 0; white-space: pre;
}
.apr-diff-block .apr-diff-line-add { color: #a6e3a1; }
.apr-diff-block .apr-diff-line-del { color: #f38ba8; }
.apr-diff-block .apr-diff-line-ctx { color: #a6adc8; }
.apr-diff-block .apr-diff-line-hdr { color: #89b4fa; font-weight: 600; }
.apr-diff-block .apr-diff-line-num { color: #585b70; margin-right: 12px; user-select: none; }
.apr-diff-block .apr-diff-issue-highlight { background: rgba(243,139,168,0.18); display: block; }

.apr-review-card {
    background: #fffbeb; border: 1px solid #fcd34d; border-radius: 8px;
    padding: 16px; margin-top: 0; margin-bottom: 16px;
}
.apr-review-card .apr-rc-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
}
.apr-review-card .apr-rc-bot-badge {
    background: #7c3aed; color: white; padding: 2px 8px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 700;
}
.apr-review-card .apr-rc-type   { font-weight: 700; color: #92400e; font-size: 0.85rem; }
.apr-review-card .apr-rc-body   { color: #44403c; font-size: 0.85rem; line-height: 1.55; }
.apr-review-card .apr-rc-suggestion {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px;
    padding: 10px 12px; margin-top: 10px;
    font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.8rem;
    color: #166534; white-space: pre-wrap;
}

.apr-model-info-banner {
    background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 8px;
    padding: 10px 14px; font-size: 0.78rem; color: #5b21b6;
    margin-top: 8px;
}
</style>
"""


# ---------------------------------------------------------------------------
# Header / banners
# ---------------------------------------------------------------------------


def render_header() -> None:
    st.markdown("# Automated Pull Request Reviewer")
    st.markdown(
        '<span style="color:#64748b;font-size:1.05rem;">'
        "Python PR review · demo UI"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="apr-badge-row">'
        '<span class="apr-badge apr-badge-blue">Python .py files only</span>'
        '<span class="apr-badge apr-badge-green">Local inference</span>'
        '<span class="apr-badge apr-badge-purple">Semantic issues</span>'
        '<span class="apr-badge apr-badge-amber">CI/CD quality gate</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_model_banner(model_info: tp.Mapping[str, tp.Any]) -> None:
    """Model line in the header strip."""
    name = html.escape(str(model_info.get("name", "(unknown)")))
    weights_path = model_info.get("weights_path", "")
    parts = [f"<strong>Model:</strong> <code>{name}</code>"]
    if weights_path:
        parts.append(
            f"&middot; <strong>weights:</strong> "
            f"<code>{html.escape(str(weights_path))}</code>",
        )
    else:
        backend = model_info.get("backend", "")
        ep = model_info.get("endpoint")
        if backend == "ollama" and ep:
            parts.append(
                f"&middot; <strong>Ollama:</strong> "
                f"<code>{html.escape(str(ep))}</code>",
            )
        elif backend == "openai" and ep:
            parts.append(
                f"&middot; <strong>API:</strong> "
                f"<code>{html.escape(str(ep))}</code>",
            )
    loaded = "loaded" if model_info.get("loaded") else "not loaded"
    parts.append(f"&middot; <strong>status:</strong> {loaded}")
    st.markdown(
        '<div class="apr-model-info-banner">' + " ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Summary cards
# ---------------------------------------------------------------------------


def _card(label: str, value_html: str) -> str:
    return (
        '<div class="apr-summary-card">'
        f'<div class="apr-summary-label">{label}</div>'
        f'<div class="apr-summary-value">{value_html}</div>'
        "</div>"
    )


def render_summary_cards(
    meta: tp.Mapping[str, tp.Any],
    n_python_files: int,
    n_blocking_issues: int,
    review_status: str,
) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            _card(
                "Repository",
                f'<span style="font-size:0.95rem;">{html.escape(str(meta.get("repo", "")))}</span>',
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _card("Pull request", f"#{html.escape(str(meta.get('pr_number', '')))}"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(_card("Python files", str(n_python_files)), unsafe_allow_html=True)
    with c4:
        color = "#ef4444" if n_blocking_issues else "#22c55e"
        st.markdown(
            _card(
                "Blocking issues",
                f'<span style="color:{color}">{n_blocking_issues}</span>',
            ),
            unsafe_allow_html=True,
        )
    with c5:
        if review_status == "changes_requested":
            badge = (
                '<span class="apr-status-changes-requested">CHANGES REQUESTED</span>'
            )
        elif review_status == "approved":
            badge = '<span class="apr-status-approved">APPROVED BY BOT</span>'
        else:
            badge = '<span class="apr-status-error">PARTIAL ANALYSIS</span>'
        st.markdown(
            f'<div class="apr-summary-card">'
            f'<div class="apr-summary-label">Review status</div>'
            f'<div class="apr-summary-value" style="margin-top:4px;">{badge}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# File sidebar
# ---------------------------------------------------------------------------


def render_file_sidebar(
    files: tp.Sequence[tp.Mapping[str, tp.Any]],
    selected_idx: int,
    state_key: str = "selected_file_idx",
) -> None:
    """Render the changed-files sidebar. Updates ``st.session_state[state_key]``."""
    st.markdown("#### Changed files")
    for idx, f in enumerate(files):
        status = f.get("status", "skipped")
        if status == "blocking_issue":
            n_iss = len(f.get("issues") or [])
            css_cls = "apr-file-item-error"
            label = f"{n_iss} blocking issue{'s' if n_iss != 1 else ''}"
        elif status == "skipped":
            css_cls = "apr-file-item-skipped"
            label = "skipped"
        elif status == "error":
            css_cls = "apr-file-item-runtime"
            label = "error"
        else:
            css_cls = "apr-file-item-clean"
            label = "clean"

        is_selected = idx == selected_idx
        border_extra = "border: 2px solid #3b82f6;" if is_selected else ""
        st.markdown(
            f'<div class="apr-file-item {css_cls}" style="{border_extra}">'
            f"<div><strong>{html.escape(str(f.get('path', '')))}</strong><br/>"
            f'<span style="font-size:0.72rem;">{label}</span></div></div>',
            unsafe_allow_html=True,
        )
        if status != "skipped":
            short_name = f.get("path", "").split("/")[-1] or f.get("path", "")
            if st.button(
                f"View {short_name}",
                key=f"file_btn_{idx}",
                use_container_width=True,
            ):
                st.session_state[state_key] = idx
                st.rerun()


# ---------------------------------------------------------------------------
# Diff and review comment rendering
# ---------------------------------------------------------------------------


def render_diff(file_data: tp.Mapping[str, tp.Any]) -> str:
    """Return GitHub-like diff HTML with optional issue-line highlighting."""
    issue_lines: tp.Set[int] = set()
    for iss in file_data.get("issues") or []:
        line_start = iss.get("line_start") or 0
        line_end = iss.get("line_end") or line_start
        for ln in range(int(line_start), int(line_end) + 1):
            issue_lines.add(ln)

    lines_html: tp.List[str] = []
    for lineno, text in file_data.get("source_lines") or []:
        escaped = html.escape(text)
        first_char = text[:1]
        if first_char == "+":
            cls = "apr-diff-line-add"
        elif first_char == "-":
            cls = "apr-diff-line-del"
        else:
            cls = "apr-diff-line-ctx"
        highlight = " apr-diff-issue-highlight" if lineno in issue_lines else ""
        lines_html.append(
            f'<span class="{cls}{highlight}">'
            f'<span class="apr-diff-line-num">{lineno}</span>{escaped}</span>',
        )

    header_text = ""
    diff = file_data.get("diff") or ""
    if diff:
        header_text = diff.split("\n", 1)[0]
    header_html = ""
    if header_text:
        header_html = (
            f'<span class="apr-diff-line-hdr">{html.escape(header_text)}</span>\n'
        )

    return (
        '<div class="apr-diff-block">' + header_html + "\n".join(lines_html) + "</div>"
    )


def render_review_comment(issue: tp.Mapping[str, tp.Any]) -> str:
    suggestion_html = ""
    suggestion = (issue.get("suggestion") or "").strip()
    if suggestion:
        suggestion_html = (
            '<div class="apr-rc-suggestion"><strong>Suggested fix:</strong>\n'
            f"{html.escape(suggestion)}</div>"
        )
    return f"""
    <div class="apr-review-card">
        <div class="apr-rc-header">
            <span class="apr-rc-bot-badge">APR Bot</span>
            <span class="apr-rc-type">{html.escape(str(issue.get('type', 'Blocking issue')))}</span>
        </div>
        <div class="apr-rc-body">{html.escape(str(issue.get('comment', '')))}</div>
        {suggestion_html}
    </div>
    """


def render_comment_action_buttons(
    issue: tp.Mapping[str, tp.Any], file_idx: int
) -> None:
    line_start = issue.get("line_start", 0)
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.button("Copy comment", key=f"copy_{file_idx}_{line_start}")
    with bc2:
        st.button("Mark useful", key=f"useful_{file_idx}_{line_start}")
    with bc3:
        st.button("Mark false positive", key=f"fp_{file_idx}_{line_start}")


def render_main_review_area(
    file_data: tp.Mapping[str, tp.Any],
    file_idx: int,
) -> None:
    path = file_data.get("path", "")
    st.markdown(f"#### `{html.escape(str(path))}`")
    additions = file_data.get("additions", 0)
    deletions = file_data.get("deletions", 0)
    lang = file_data.get("language", "python")
    st.markdown(
        f'<span class="apr-badge apr-badge-blue">{html.escape(lang)}</span>'
        f'<span class="apr-badge apr-badge-green">+{additions} / -{deletions}</span>',
        unsafe_allow_html=True,
    )

    status = file_data.get("status", "")
    if status == "skipped":
        msg = (
            file_data.get("error")
            or "This file was skipped (non-Python or out of demo budget)."
        )
        st.info(msg)
        return
    if status == "error":
        st.error(
            f"Could not analyze this file: {file_data.get('error') or 'unknown error'}"
        )
        if file_data.get("source_lines"):
            st.markdown(render_diff(file_data), unsafe_allow_html=True)
        return

    if file_data.get("source_lines"):
        st.markdown(render_diff(file_data), unsafe_allow_html=True)
    elif file_data.get("diff"):
        st.code(file_data["diff"], language="diff")
    else:
        st.info("No diff available for this file.")
        return

    issues = file_data.get("issues") or []
    if not issues:
        st.markdown(
            '<div style="background:#f0fdf4;border:1px solid #bbf7d0;'
            "border-radius:8px;padding:14px;color:#166534;font-size:0.85rem;"
            'margin-top:10px;">'
            "&#10003; &nbsp;No blocking issues found in this file."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for iss in issues:
        st.markdown(render_review_comment(iss), unsafe_allow_html=True)
        render_comment_action_buttons(iss, file_idx)


# ---------------------------------------------------------------------------
# Context expanders
# ---------------------------------------------------------------------------


def render_context_section(context: tp.Mapping[str, tp.Any]) -> None:
    st.markdown("### Context used by model")
    with st.expander("PR title and description"):
        st.markdown(f"**{html.escape(str(context.get('pr_title', '')))}**")
        body = context.get("pr_body") or ""
        st.markdown(body or "_(empty)_")
    with st.expander("Changed file context"):
        st.code(context.get("changed_file_context", "Not available"), language="text")
    with st.expander("Imported definitions (outgoing dependencies)"):
        for defn in context.get("imported_definitions") or []:
            st.code(defn, language="text")
    with st.expander("Usage sites (incoming dependencies)"):
        for site in context.get("usage_sites") or []:
            st.code(site, language="text")
    with st.expander("Repository metadata summary"):
        st.code(context.get("repo_metadata", "Not available"), language="text")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "CUSTOM_CSS",
    "render_comment_action_buttons",
    "render_context_section",
    "render_diff",
    "render_file_sidebar",
    "render_header",
    "render_main_review_area",
    "render_model_banner",
    "render_review_comment",
    "render_summary_cards",
]
