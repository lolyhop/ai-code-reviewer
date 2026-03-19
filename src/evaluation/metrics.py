from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from bert_score import score as bertscore_score


@dataclass
class CommentEvaluationResult:
    """
    Aggregated comment evaluation metrics.
    Values are means over all input pairs.
    """

    mean_precision: float
    mean_recall: float
    mean_f1: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
        }


class CommentEvaluator:
    """
    Compute BERTScore-style metrics for pairs of human and AI-generated comments.

    By default, this uses the `microsoft/codebert-base` model to obtain
    code-aware contextual embeddings, effectively giving a CodeBERTScore-style
    evaluation for review comments that may contain code tokens.
    """

    def __init__(
        self,
        model_type: str = "microsoft/codebert-base",
        num_layers: Optional[int] = 12,
        batch_size: int = 16,
        device: Optional[str] = None,
        verbose: bool = False,
        idf: bool = False,
        lang: Optional[str] = None,
    ) -> None:
        """
        Initialize the evaluator.

        Parameters
        ----------
        model_type:
            Hugging Face model identifier to use for embeddings.
            Default: "microsoft/codebert-base".
        num_layers:
            Number of layers to use from the model. If None, use library default.
        batch_size:
            Batch size used internally by bert-score.
        device:
            PyTorch device string (e.g., "cuda", "cuda:0", "cpu"). If None,
            `bert_score` chooses automatically.
        verbose:
            Whether to print progress from `bert_score`.
        idf:
            Whether to use IDF weighting in BERTScore.
        lang:
            Optional language code for some `bert_score` optimizations.
            For code/mixed comments, this can be left as None.
        """
        self.model_type = model_type
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.idf = idf
        self.lang = lang

    def evaluate(
        self,
        references: List[str],
        candidates: List[str],
    ) -> CommentEvaluationResult:
        """
        Evaluate AI-generated comments against human reference comments.

        Parameters
        ----------
        references:
            A list of human reference comments (ground truth),
            one per example.
        candidates:
            A list of AI-generated comments, aligned by index with `references`.

        Returns
        -------
        CommentEvaluationResult
            Aggregated mean Precision, Recall, and F1 scores over all examples.

        Raises
        ------
        ValueError
            If lengths of `references` and `candidates` do not match,
            or if input sequences are empty.
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"references and candidates must have the same length; "
                f"got {len(references)} and {len(candidates)}"
            )
        if len(references) == 0:
            raise ValueError("references and candidates must be non-empty.")

        P, R, F1 = bertscore_score(
            cands=list(candidates),
            refs=list(references),
            model_type=self.model_type,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
            idf=self.idf,
            lang=self.lang,
        )

        mean_precision: float = float(P.mean().item())
        mean_recall: float = float(R.mean().item())
        mean_f1: float = float(F1.mean().item())

        return CommentEvaluationResult(
            mean_precision=mean_precision,
            mean_recall=mean_recall,
            mean_f1=mean_f1,
        )

    def evaluate_to_dict(
        self,
        references: List[str],
        candidates: List[str],
    ) -> Dict[str, float]:
        """
        Convenience wrapper returning metrics as a plain dictionary.

        Keys:
            - "mean_precision"
            - "mean_recall"
            - "mean_f1"
        """
        result = self.evaluate(references=references, candidates=candidates)
        return result.to_dict()


if __name__ == "__main__":
    human_comments = [
        "This function does not handle the None case for the input parameter.",
        "Consider extracting this nested loop into a helper to improve readability.",
        "The variable name is misleading; it represents a count, not an index.",
    ]

    ai_comments = [
        "Edge case when the argument is None is not covered here.",
        "You could move this inner loop into a separate function for clarity.",
        "This variable name suggests an index but it's actually used as a counter.",
    ]

    evaluator = CommentEvaluator(
        model_type="microsoft/codebert-base",
        num_layers=12,
        batch_size=8,
        verbose=True,
    )

    metrics = evaluator.evaluate_to_dict(
        references=human_comments,
        candidates=ai_comments,
    )

    print("Smoke test for microsoft/codebert-base")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
