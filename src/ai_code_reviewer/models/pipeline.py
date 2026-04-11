import json
import pandas as pd
import typing as tp

from src.ai_code_reviewer.data_processing.data_cleaning import (
    truncate_file_tree,
    preprocess_patched,
    _format_patched_content,
    _format_dependencies,
)
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
        file_tree_max_depth: int = 2,
    ) -> None:
        self.retriever = Retriever(retriever_type=retriever_type, top_k=top_k)
        self.file_tree_max_depth = file_tree_max_depth

    def run(self, data: tp.Any) -> tp.Dict[str, tp.Any]:
        """Run the review pipeline on input data."""

        samples = self._process_input_data(data)

        incoming = self.retriever.retrieve(samples, retrieval_field="incoming_dependencies")
        outgoing = self.retriever.retrieve(samples, retrieval_field="outgoing_dependencies")

        repo_metadata = [build_repository_metadata(s) for s in samples]
        pr_metadata = [build_pull_request_metadata(s) for s in samples]
        file_trees = [
            truncate_file_tree(s.file_tree or "", self.file_tree_max_depth)
            for s in samples
        ]

        prompts = [
            self._build_prompt(
                sample=samples[i],
                repo_meta=repo_metadata[i],
                pr_meta=pr_metadata[i],
                file_tree=file_trees[i],
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
            "file_trees": file_trees,
            "prompts": prompts,
        }

    @staticmethod
    def _build_prompt(
        sample: ReviewSample,
        repo_meta: str,
        pr_meta: str,
        file_tree: str,
        incoming_deps: tp.Dict[str, str],
        outgoing_deps: tp.Dict[str, str],
    ) -> str:
        numbered = preprocess_patched(sample.patched_content or "")
        return REVIEWER_PROMPT_TEMPLATE.format(
            repository_metadata=repo_meta,
            pull_request_metadata=pr_meta,
            file_tree=file_tree or "Not available",
            patched_content=_format_patched_content(sample.path, numbered),
            incoming_dependencies=_format_dependencies(incoming_deps),
            outgoing_dependencies=_format_dependencies(outgoing_deps),
        )

    def unpack_review_sample(
        self, row: tp.Union[str, tp.Dict[str, tp.Any]]
    ) -> ReviewSample:
        """Convert row data to ReviewSample object."""
        payload = self._ensure_dict(row)
        return ReviewSample.from_dict(payload)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="ReviewPipeline runner")
    parser.add_argument(
        "--infer", action="store_true", help="Run model inference on generated prompts"
    )
    parser.add_argument("--top-k", type=int, default=3, help="Retriever top_k")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    args = parser.parse_args()

    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "raw_dataset.json"
    )
    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    if args.max_samples:
        raw_data = raw_data[: args.max_samples]

    pipeline = ReviewPipeline(retriever_type="heuristic", top_k=args.top_k)
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

    if args.infer:
        from src.ai_code_reviewer.models.inference import ReviewModel

        model = ReviewModel()
        model.load()
        predictions = model.predict_batch(prompts)

        for i, pred in enumerate(predictions):
            print(f"\n{'=' * 80}")
            print(f"Prediction {i} | {samples[i].repo} | {samples[i].path}")
            print(f"Issues found: {len(pred.issues)}")
            for j, issue in enumerate(pred.issues):
                print(
                    f"  [{j}] L{issue.line_start}-L{issue.line_end}: {issue.comment}"
                )
