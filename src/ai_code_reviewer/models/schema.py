from dataclasses import dataclass, field
import typing as tp


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
    comments: tp.List[tp.Dict[str, tp.Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: tp.Dict[str, tp.Any]) -> "ReviewSample":
        return cls(
            repo=payload.get("repo", ""),
            pr_number=int(payload.get("pr_number", 0)),
            pr_title=payload.get("pr_title", ""),
            pr_body=payload.get("pr_body", ""),
            repo_star_count=int(payload.get("repo_star_count", 0)),
            commit_sha=payload.get("commit_sha", ""),
            path=payload.get("path", ""),
            patched_content=payload.get("patched_content", ""),
            outgoing_dependencies=payload.get("outgoing_dependencies") or {},
            incoming_dependencies=payload.get("incoming_dependencies") or {},
            metadata_files=payload.get("metadata_files") or {},
            comments=payload.get("comments") or [],
        )


@dataclass
class PredictedIssue:
    line_start: tp.Optional[int]
    line_end: tp.Optional[int]
    comment: str


@dataclass
class ReviewPrediction:
    issues: tp.List[PredictedIssue] = field(default_factory=list)
