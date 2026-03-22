from __future__ import annotations


def normalize_repo_rel_path(path: str) -> str | None:
    """Return repo-relative POSIX path with '..' and '.' resolved, or None if unsafe.

    Args:
        path:
            Raw path from GitHub or archives.

    Returns:
        Normalized path, or None if empty or outside repo root.
    """
    if not path or "\x00" in path:
        return None
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    parts: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts) if parts else None
