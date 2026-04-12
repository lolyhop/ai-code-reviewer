from dataclasses import dataclass, field
import typing as tp

import pandas as pd


def _get(payload: tp.Dict[str, tp.Any], key: str, default: tp.Any) -> tp.Any:
    """Safely get a value from *payload*, returning *default* only when the
    value is ``None`` (avoids calling ``bool()`` on numpy arrays / pandas
    objects which would raise ``ValueError``).
    """
    val = payload.get(key)
    if val is None:
        return default
    return val


def _to_path_content_dict(
    raw: tp.Any,
    path_keys: tp.Tuple[str, ...] = ("path", "file_path"),
    content_keys: tp.Tuple[str, ...] = ("patched_content", "content"),
) -> tp.Dict[str, str]:
    """Normalize dependency/metadata arrays into ``{path: content}`` dicts."""
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
        if not isinstance(item, dict):
            continue
        path = ""
        for k in path_keys:
            if k in item:
                path = item[k]
                break
        if not path:
            path = f"item_{len(out)}"
        content = ""
        for k in content_keys:
            if k in item:
                content = item[k]
                break
        out[path] = content
    return out


@dataclass
class ReviewSample:
    repo: str
    pr_number: int
    pr_title: str
    pr_body: str
    repo_star_count: int
    commit_sha: str
    path: str
    patched_content: str
    outgoing_dependencies: tp.Dict[str, str] = field(default_factory=dict)
    incoming_dependencies: tp.Dict[str, str] = field(default_factory=dict)
    metadata_files: tp.Dict[str, str] = field(default_factory=dict)
    file_tree: str = ""
    comments: tp.List[tp.Dict[str, tp.Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: tp.Dict[str, tp.Any]) -> "ReviewSample":
        return cls(
            repo=_get(payload, "repo", ""),
            pr_number=int(_get(payload, "pr_number", 0)),
            pr_title=_get(payload, "pr_title", ""),
            pr_body=_get(payload, "pr_body", ""),
            repo_star_count=int(_get(payload, "repo_star_count", 0)),
            commit_sha=_get(payload, "commit_sha", ""),
            path=_get(payload, "path", ""),
            patched_content=_get(payload, "patched_content", ""),
            outgoing_dependencies=_to_path_content_dict(
                _get(payload, "outgoing_dependencies", {})
            ),
            incoming_dependencies=_to_path_content_dict(
                _get(payload, "incoming_dependencies", {})
            ),
            metadata_files=_to_path_content_dict(_get(payload, "metadata_files", {})),
            file_tree=_get(payload, "file_tree", ""),
            comments=_get(payload, "comments", []),
        )


@dataclass
class PredictedIssue:
    line_start: tp.Optional[int]
    line_end: tp.Optional[int]
    comment: str


@dataclass
class ReviewPrediction:
    issues: tp.List[PredictedIssue] = field(default_factory=list)
