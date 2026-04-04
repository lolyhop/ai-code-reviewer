import os
import re
import typing as tp

from src.ai_code_reviewer.models.schema import ReviewSample

DEPENDENCY_BASENAMES = frozenset(
    {
        "requirements.txt",
        "pyproject.toml",
        "Pipfile",
        "setup.cfg",
        "environment.yml",
    }
)

README_BASENAMES = frozenset(
    {
        "README.md",
        "README.rst",
        "README.txt",
        "README",
    }
)

CONTRIBUTING_BASENAMES = frozenset(
    {
        "CONTRIBUTING.md",
        "CONTRIBUTING.rst",
        "CONTRIBUTING.txt",
        "CONTRIBUTING",
    }
)

CONTRIBUTING_MAX_CHARS = 400

README_MAX_CHARS = 400
SETUP_PY_MAX_CHARS = 500
METADATA_MAX_TOTAL_CHARS = 1500
PR_BODY_MAX_CHARS = 500
SECTION_SEPARATOR = "\n\n"


def _is_dependency_file(path: str) -> bool:
    return os.path.basename(path) in DEPENDENCY_BASENAMES


def _is_readme(path: str) -> bool:
    return os.path.basename(path).upper().split(".")[0] == "README"


def _is_contributing(path: str) -> bool:
    return os.path.basename(path).upper().split(".")[0] == "CONTRIBUTING"


def _is_setup_py(path: str) -> bool:
    return os.path.basename(path) == "setup.py"


def _truncate_readme(content: str, max_chars: int = README_MAX_CHARS) -> str:
    if len(content) <= max_chars:
        return content
    cut = content[:max_chars]
    last_newline = cut.rfind("\n")
    if last_newline > max_chars // 2:
        cut = cut[:last_newline]
    return cut.rstrip() + "\n[truncated]"


_INSTALL_REQUIRES_RE = re.compile(r"install_requires\s*=\s*\[([^\]]*)\]", re.DOTALL)
_EXTRAS_REQUIRE_RE = re.compile(r"extras_require\s*=\s*\{([^}]*)\}", re.DOTALL)
_NAME_RE = re.compile(r"""name\s*=\s*['"]([^'"]+)['"]""")
_VERSION_RE = re.compile(r"""version\s*=\s*['"]([^'"]+)['"]""")


def _extract_setup_py(content: str) -> str:
    """Extract dependency and identity info from setup.py."""
    parts: tp.List[str] = []

    name_m = _NAME_RE.search(content)
    version_m = _VERSION_RE.search(content)
    if name_m:
        parts.append(f"name: {name_m.group(1)}")
    if version_m:
        parts.append(f"version: {version_m.group(1)}")

    install_m = _INSTALL_REQUIRES_RE.search(content)
    if install_m:
        parts.append(f"install_requires: [{install_m.group(1).strip()}]")

    extras_m = _EXTRAS_REQUIRE_RE.search(content)
    if extras_m:
        parts.append(f"extras_require: {{{extras_m.group(1).strip()}}}")

    if parts:
        return "\n".join(parts)

    if len(content) <= SETUP_PY_MAX_CHARS:
        return content
    return content[:SETUP_PY_MAX_CHARS].rstrip() + "\n[truncated]"


def _append_within_budget(
    sections: tp.List[str],
    new_section: str,
    current_len: int,
    budget: int,
) -> int:
    """Append *new_section* if it fits (fully or truncated) within *budget*.

    Returns updated current_len.
    """
    sep_len = len(SECTION_SEPARATOR) if sections else 0
    available = budget - current_len - sep_len
    if available <= 0:
        return current_len

    if len(new_section) <= available:
        sections.append(new_section)
        return current_len + sep_len + len(new_section)

    truncated = new_section[:available].rstrip()
    if truncated:
        sections.append(truncated + "\n[truncated]")
    return budget


def summarize_metadata_files(
    metadata_files: tp.Dict[str, str],
    max_total_chars: int = METADATA_MAX_TOTAL_CHARS,
) -> str:
    """Heuristic compression of repository metadata files.

    Sections are added by priority until *max_total_chars* is reached:
      1. README — truncated to first ~400 chars.
      2. CONTRIBUTING — truncated to first ~400 chars.
      3. Dependency files (requirements.txt, pyproject.toml, etc.) — kept as-is.
      4. setup.py — extract name/version/install_requires.
      5. Everything else — skipped.
    """
    readme_files: tp.List[tp.Tuple[str, str]] = []
    contributing_files: tp.List[tp.Tuple[str, str]] = []
    dep_files: tp.List[tp.Tuple[str, str]] = []
    setup_files: tp.List[tp.Tuple[str, str]] = []

    for path, content in metadata_files.items():
        if _is_readme(path):
            readme_files.append((path, content))
        elif _is_contributing(path):
            contributing_files.append((path, content))
        elif _is_dependency_file(path):
            dep_files.append((path, content))
        elif _is_setup_py(path):
            setup_files.append((path, content))

    sections: tp.List[str] = []
    used = 0

    for path, content in readme_files:
        remaining = max_total_chars - used - (len(SECTION_SEPARATOR) if sections else 0)
        budget = min(README_MAX_CHARS, max(remaining, 0))
        if budget > 0:
            used = _append_within_budget(
                sections,
                f"[{path}]\n{_truncate_readme(content, max_chars=budget)}",
                used,
                max_total_chars,
            )

    for path, content in contributing_files:
        remaining = max_total_chars - used - (len(SECTION_SEPARATOR) if sections else 0)
        budget = min(CONTRIBUTING_MAX_CHARS, max(remaining, 0))
        if budget > 0:
            used = _append_within_budget(
                sections,
                f"[{path}]\n{_truncate_readme(content, max_chars=budget)}",
                used,
                max_total_chars,
            )

    for path, content in dep_files:
        used = _append_within_budget(
            sections, f"[{path}]\n{content.strip()}", used, max_total_chars
        )

    for path, content in setup_files:
        used = _append_within_budget(
            sections, f"[{path}]\n{_extract_setup_py(content)}", used, max_total_chars
        )

    return SECTION_SEPARATOR.join(sections) if sections else ""


def _truncate_on_line_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars // 2:
        cut = cut[:last_nl]
    return cut.rstrip() + "\n[truncated]"


def _strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def build_repository_metadata(
    sample: ReviewSample,
    max_total_chars: int = METADATA_MAX_TOTAL_CHARS,
) -> str:
    """Build an LLM-friendly repository context string."""
    lines = [
        f"This code belongs to the GitHub repository '{sample.repo}'.",
    ]
    if sample.repo_star_count:
        lines.append(f"The repository has {sample.repo_star_count} stars.")

    header = " ".join(lines)
    header_overhead = len(header) + len(SECTION_SEPARATOR)
    files_budget = max(max_total_chars - header_overhead, 0)
    files_summary = summarize_metadata_files(
        sample.metadata_files, max_total_chars=files_budget
    )

    if files_summary:
        return f"{header}\n\n{files_summary}"
    return header


def build_pull_request_metadata(
    sample: ReviewSample,
    body_max_chars: int = PR_BODY_MAX_CHARS,
) -> str:
    """Build a concise PR metadata string from title and body."""
    parts = [f"PR title: {sample.pr_title}"]

    body = (sample.pr_body or "").strip()
    if body:
        body = _strip_html_tags(body)
        body = _truncate_on_line_boundary(body, body_max_chars)
        parts.append(f"PR description:\n{body}")

    return "\n".join(parts)
