import json
import pandas as pd
import typing as tp

from src.ai_code_reviewer.models.metadata import (
    build_pull_request_metadata,
    build_repository_metadata,
)
from src.ai_code_reviewer.models.prompts import REVIEWER_PROMPT_TEMPLATE
from src.ai_code_reviewer.models.retriever import Retriever
from src.ai_code_reviewer.models.schema import ReviewSample


class DatasetRowError(ValueError):
    """Exception raised when dataset row processing fails."""

    pass


class ReviewPipeline:
    """Pipeline for processing code review samples."""

    def _ensure_dict(
        self, row: tp.Union[str, tp.Dict[str, tp.Any]]
    ) -> tp.Dict[str, tp.Any]:
        """Convert row data to dictionary format."""
        if isinstance(row, dict):
            return row

        if isinstance(row, str):
            try:
                payload = json.loads(row)
            except json.JSONDecodeError as exc:
                raise DatasetRowError(
                    f"Failed to parse dataset row as JSON: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise DatasetRowError("Parsed dataset row is not a JSON object.")

            return payload

        raise DatasetRowError(
            f"Unsupported row type: {type(row).__name__}. Expected str or dict."
        )

    def _process_input_data(self, data: tp.Any) -> tp.List[ReviewSample]:
        """Process input data into a list of ReviewSample objects."""
        if isinstance(data, pd.DataFrame):
            return [self.unpack_review_sample(row) for _, row in data.iterrows()]

        if isinstance(data, list):
            return [self.unpack_review_sample(row) for row in data]

        return [self.unpack_review_sample(data)]

    def __init__(
        self,
        retriever_type: str = "heuristic",
        top_k: int = 3,
    ) -> None:
        self.retriever = Retriever(retriever_type=retriever_type, top_k=top_k)

    def run(self, data: tp.Any) -> tp.Dict[str, tp.Any]:
        """Run the review pipeline on input data."""
        
        # 1. Process the input data
        samples = self._process_input_data(data)

        # 2. Retrieve top-k relevant dependencies for each sample
        incoming = self.retriever.retrieve(samples, retrieval_field="incoming_dependencies")
        outgoing = self.retriever.retrieve(samples, retrieval_field="outgoing_dependencies")

        # 3. Prepare repository metadata
        repo_metadata = [build_repository_metadata(s) for s in samples]

        # 4. Prepare Pull Request metadata
        pr_metadata = [build_pull_request_metadata(s) for s in samples]

        # 5. Build prompts
        prompts = [
            self._build_prompt(
                sample=samples[i],
                repo_meta=repo_metadata[i],
                pr_meta=pr_metadata[i],
                incoming_deps=incoming[i],
                outgoing_deps=outgoing[i],
            )
            for i in range(len(samples))
        ]

        return {
            "samples": samples,
            "incoming": incoming,
            "outgoing": outgoing,
            "repo_metadata": repo_metadata,
            "pr_metadata": pr_metadata,
            "prompts": prompts,
        }

    @staticmethod
    def _format_patched_content(sample: ReviewSample) -> str:
        """Add file path header and line numbers to patched content."""
        lines = (sample.patched_content or "").splitlines()
        numbered = [f"{i + 1:>4} | {line}" for i, line in enumerate(lines)]
        return f"File: {sample.path}\n" + "\n".join(numbered)

    @staticmethod
    def _format_dependencies(deps: tp.Dict[str, str]) -> str:
        if not deps:
            return "None"
        sections = []
        for path, content in deps.items():
            sections.append(f"--- {path} ---\n{content.strip()}")
        return "\n\n".join(sections)

    @staticmethod
    def _build_prompt(
        sample: ReviewSample,
        repo_meta: str,
        pr_meta: str,
        incoming_deps: tp.Dict[str, str],
        outgoing_deps: tp.Dict[str, str],
    ) -> str:
        return REVIEWER_PROMPT_TEMPLATE.format(
            repository_metadata=repo_meta,
            pull_request_metadata=pr_meta,
            patched_content=ReviewPipeline._format_patched_content(sample),
            incoming_dependencies=ReviewPipeline._format_dependencies(incoming_deps),
            outgoing_dependencies=ReviewPipeline._format_dependencies(outgoing_deps),
        )

    def unpack_review_sample(
        self, row: tp.Union[str, tp.Dict[str, tp.Any]]
    ) -> ReviewSample:
        """Convert row data to ReviewSample object."""
        payload = self._ensure_dict(row)
        return ReviewSample.from_dict(payload)


if __name__ == "__main__":
    import os

    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "raw_dataset.json"
    )
    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    pipeline = ReviewPipeline(retriever_type="heuristic", top_k=3)
    result = pipeline.run(raw_data)

    samples = result["samples"]
    prompts = result["prompts"]

    print(f"Total samples: {len(samples)}\n")

    for i, sample in enumerate(samples):
        print(f"{'=' * 80}")
        print(f"Sample {i} | {sample.repo} | {sample.path}")
        print(f"Prompt length: {len(prompts[i])} chars")
        print(f"{'=' * 80}")
        print(prompts[i][:2000])
        if len(prompts[i]) > 2000:
            print("\n... [prompt truncated for display] ...")
        print()
