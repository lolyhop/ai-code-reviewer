import os
import typing as tp

from src.ai_code_reviewer.models.schema import ReviewSample

JUNK_BASENAMES = frozenset({"__init__.py", "setup.py", "conftest.py", "__version__.py"})


class Retriever:
    """Select top-k dependency snippets per sample."""

    SUPPORTED_TYPES = {"heuristic"}

    def __init__(
        self,
        retriever_type: str = "heuristic",
        top_k: int = 3,
    ) -> None:
        if retriever_type not in self.SUPPORTED_TYPES:
            supported = ", ".join(sorted(self.SUPPORTED_TYPES))
            raise ValueError(
                f"Unsupported retriever_type: {retriever_type}. "
                f"Supported values: {supported}."
            )

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")

        self.retriever_type = retriever_type
        self.top_k = top_k

    @staticmethod
    def _validate_field(retrieval_field: str) -> None:
        sample_fields = {f.name for f in ReviewSample.__dataclass_fields__.values()}
        if retrieval_field not in sample_fields:
            raise ValueError(
                f"retrieval_field '{retrieval_field}' does not exist on ReviewSample. "
                f"Available fields: {', '.join(sorted(sample_fields))}."
            )

    def retrieve(
        self, items: tp.List[ReviewSample], retrieval_field: str
    ) -> tp.List[tp.Dict[str, str]]:
        """Return filtered top-k dependency dicts, one per input sample."""
        self._validate_field(retrieval_field)

        if self.retriever_type == "heuristic":
            return [
                self._heuristic_retrieve(sample, retrieval_field) for sample in items
            ]

        raise ValueError(f"Unsupported retriever_type: {self.retriever_type}.")

    def _heuristic_retrieve(
        self, sample: ReviewSample, retrieval_field: str
    ) -> tp.Dict[str, str]:
        """Filter out junk files and return the first top_k entries."""
        deps: tp.Dict[str, str] = getattr(sample, retrieval_field, {})

        filtered = {
            path: content
            for path, content in deps.items()
            if os.path.basename(path) not in JUNK_BASENAMES
        }

        return dict(list(filtered.items())[: self.top_k])
