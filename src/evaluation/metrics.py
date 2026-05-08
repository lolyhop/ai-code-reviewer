"""
Metrics for comparing `target` vs `prediction` JSON (review issues with line ranges).

Supports CSV exports from the review pipeline with columns ``target`` and ``prediction``.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import pandas as pd
from bert_score import BERTScorer

logger = logging.getLogger(__name__)

IssueNorm = Tuple[int, int, str]  # start_line, end_line, comment


class MetricsReport(TypedDict, total=False):
    """Structured report written to JSON."""

    binary: Dict[str, Any]
    line_range_iou: Dict[str, Any]
    bertscore: Dict[str, Any]
    bertscore_per_prediction: Dict[str, Any]
    n_rows: int
    bertscore_skipped: str


@dataclass
class CommentEvaluationResult:
    """Aggregated BERTScore-style metrics (means over input pairs)."""

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
    BERTScore for pairs of human vs model comments (CodeBERT-style embeddings by default).
    """

    def __init__(
        self,
        model_type: str = "microsoft/codebert-base",
        num_layers: Optional[int] = 12,
        batch_size: int = 16,
        device: Optional[str] = None,
        idf: bool = False,
        lang: Optional[str] = None,
    ) -> None:
        self.scorer = BERTScorer(
            model_type=model_type,
            num_layers=num_layers,
            batch_size=batch_size,
            device=device,
            idf=idf,
            lang=lang,
        )

    def evaluate(
        self,
        references: List[str],
        candidates: List[str],
    ) -> CommentEvaluationResult:
        if len(references) != len(candidates):
            raise ValueError(
                f"references and candidates must have the same length; "
                f"got {len(references)} and {len(candidates)}"
            )
        if len(references) == 0:
            raise ValueError("references and candidates must be non-empty.")

        P, R, F1 = self.scorer.score(
            cands=list(candidates),
            refs=list(references),
        )

        return CommentEvaluationResult(
            mean_precision=float(P.mean().item()),
            mean_recall=float(R.mean().item()),
            mean_f1=float(F1.mean().item()),
        )

    def evaluate_to_dict(
        self,
        references: List[str],
        candidates: List[str],
    ) -> Dict[str, float]:
        return self.evaluate(references=references, candidates=candidates).to_dict()

    def score_aligned_pairs(
        self,
        references: List[str],
        candidates: List[str],
        chunk_size: int = 32,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Return per-pair precision, recall, F1 (same length as inputs)."""
        if len(references) != len(candidates):
            raise ValueError("references and candidates length mismatch")
        if not references:
            return [], [], []
        p_all: List[float] = []
        r_all: List[float] = []
        f_all: List[float] = []
        for start in range(0, len(references), chunk_size):
            rc = references[start : start + chunk_size]
            cd = candidates[start : start + chunk_size]
            P, R, F1 = self.scorer.score(cands=cd, refs=rc)
            p_all.extend(float(x) for x in np.asarray(P).flatten())
            r_all.extend(float(x) for x in np.asarray(R).flatten())
            f_all.extend(float(x) for x in np.asarray(F1).flatten())
        return p_all, r_all, f_all


def _parse_cell(raw: Any) -> Any:
    """Parse a CSV cell into a Python object (JSON, Python literal, or dict)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                logger.warning(
                    "Could not parse cell as JSON or literal: %s...", s[:120]
                )
                return {}
    return {}


def _normalize_issue(issue: Dict[str, Any]) -> Optional[IssueNorm]:
    """One issue -> (start, end, comment) with inclusive line indices."""
    if not isinstance(issue, dict):
        return None
    start: Optional[int] = None
    end: Optional[int] = None
    if "line_range" in issue and isinstance(issue["line_range"], dict):
        lr = issue["line_range"]
        start = lr.get("start")
        end = lr.get("end")
    else:
        start = issue.get("line_start")
        end = issue.get("line_end")
    if start is None or end is None:
        return None
    try:
        s_int = int(start)
        e_int = int(end)
    except (TypeError, ValueError):
        return None
    comment = str(issue.get("comment", "")).strip()
    return (s_int, e_int, comment)


def parse_issues_payload(payload: Any) -> List[IssueNorm]:
    """Extract normalized issues from parsed JSON / dict."""
    if not isinstance(payload, dict):
        return []
    issues_raw = payload.get("issues", [])
    if not isinstance(issues_raw, list):
        return []
    out: List[IssueNorm] = []
    for it in issues_raw:
        if not isinstance(it, dict):
            continue
        n = _normalize_issue(it)
        if n is not None:
            out.append(n)
    return out


def line_span_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """IoU over inclusive integer line ranges [start, end]."""
    inter = max(0, min(end1, end2) - max(start1, start2) + 1)
    len1 = end1 - start1 + 1
    len2 = end2 - start2 + 1
    union = len1 + len2 - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def mean_greedy_iou_and_pairs(
    gt: Sequence[IssueNorm], pred: Sequence[IssueNorm]
) -> Tuple[float, List[Tuple[IssueNorm, IssueNorm]]]:
    """
    Greedy match: each GT issue gets at most one predicted issue (highest IoU first pass).

    Returns mean IoU over **ground-truth** issues (unmatched GT count as IoU 0), and
    matched pairs for BERTScore (GT comment vs pred comment).
    """
    if not gt and not pred:
        return 1.0, []
    if not gt:
        return 0.0, []
    if not pred:
        return 0.0, []

    pred_spans = [(p[0], p[1]) for p in pred]
    gt_spans = [(g[0], g[1]) for g in gt]
    used_pred: set[int] = set()
    ious: List[float] = []
    pairs: List[Tuple[IssueNorm, IssueNorm]] = []

    for gi, g in enumerate(gt):
        best_j = -1
        best_iou = -1.0
        gs, ge = gt_spans[gi]
        for j, (ps, pe) in enumerate(pred_spans):
            if j in used_pred:
                continue
            iou = line_span_iou(gs, ge, ps, pe)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used_pred.add(best_j)
            ious.append(best_iou)
            pairs.append((g, pred[best_j]))
        else:
            ious.append(0.0)

    return float(sum(ious) / len(ious)), pairs


def _safe_int_label(val: Any) -> int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def binary_classification_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, Any]:
    """Accuracy, precision/recall/F1 for positive class (1 = blocking issue present)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    n = len(y_true)
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision_class_1_blocking": precision,
        "recall_class_1_blocking": recall,
        "f1_class_1_blocking": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {
            "n_total": n,
            "n_positive_label": tp + fn,
            "n_negative_label": tn + fp,
        },
    }


def compute_bertscore_per_prediction_max_over_refs(
    df: pd.DataFrame,
    comment_evaluator: CommentEvaluator,
) -> Dict[str, Any]:
    """
    For each predicted issue in a row, take the **best** BERTScore match among all
    human issues in that row (argmax F1). P/R are taken from that same pair.

    Rows with predicted issues but no human issues in ``target`` get (0, 0, 0).

    Aggregates **mean** and **median** over all predicted issues (model-centric).
    """
    pair_refs: List[str] = []
    pair_cands: List[str] = []
    pair_key: List[Tuple[int, int]] = []

    best: Dict[Tuple[int, int], Tuple[float, float, float]] = {}

    for pos, (_, row) in enumerate(df.iterrows()):
        t_payload = _parse_cell(row["target"])
        p_payload = _parse_cell(row["prediction"])
        gt_issues = parse_issues_payload(t_payload)
        pr_issues = parse_issues_payload(p_payload)

        for j, pr in enumerate(pr_issues):
            key = (pos, j)
            p_txt = pr[2].strip() if pr[2] else ""
            cand = p_txt if p_txt else " "
            if not gt_issues:
                best[key] = (0.0, 0.0, 0.0)
                continue
            for g in gt_issues:
                g_txt = g[2].strip() if g[2] else ""
                ref = g_txt if g_txt else " "
                pair_refs.append(ref)
                pair_cands.append(cand)
                pair_key.append(key)

    if pair_key:
        P_list, R_list, F_list = comment_evaluator.score_aligned_pairs(
            pair_refs, pair_cands
        )
        for i, key in enumerate(pair_key):
            f1v, pv, rv = F_list[i], P_list[i], R_list[i]
            cur = best.get(key)
            if cur is None or f1v > cur[0]:
                best[key] = (f1v, pv, rv)

    per_f1: List[float] = []
    per_p: List[float] = []
    per_r: List[float] = []

    for pos, (_, row) in enumerate(df.iterrows()):
        p_payload = _parse_cell(row["prediction"])
        pr_issues = parse_issues_payload(p_payload)
        for j in range(len(pr_issues)):
            key = (pos, j)
            f1v, pv, rv = best.get(key, (0.0, 0.0, 0.0))
            per_f1.append(f1v)
            per_p.append(pv)
            per_r.append(rv)

    n = len(per_f1)
    if n == 0:
        return {
            "n_model_issues": 0,
            "mean_f1": 0.0,
            "median_f1": 0.0,
            "mean_precision": 0.0,
            "median_precision": 0.0,
            "mean_recall": 0.0,
            "median_recall": 0.0,
            "description": "Per predicted issue: max BERT F1 vs any human issue in row.",
        }

    return {
        "n_model_issues": n,
        "mean_f1": float(sum(per_f1) / n),
        "median_f1": float(statistics.median(per_f1)),
        "mean_precision": float(sum(per_p) / n),
        "median_precision": float(statistics.median(per_p)),
        "mean_recall": float(sum(per_r) / n),
        "median_recall": float(statistics.median(per_r)),
        "description": "Per predicted issue: max BERT F1 vs any human issue in the same row; "
        "P/R from that pair. Averaged over all model issues.",
    }


def evaluate_predictions_dataframe(
    df: pd.DataFrame,
    comment_evaluator: Optional[CommentEvaluator] = None,
    skip_bertscore: bool = False,
) -> MetricsReport:
    """
    Compute metrics from a dataframe with ``target`` and ``prediction`` columns.

    Binary task: ``label`` (0/1) vs predicted positive = any predicted issue.
    IoU: greedy match on line ranges; mean over GT issues per row, then mean over rows.
    BERTScore: on (gt_comment, pred_comment) for greedy-matched pairs (non-empty texts).
    """
    required = {"target", "prediction", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    y_true: List[int] = []
    y_pred_bin: List[int] = []
    iou_row_means: List[float] = []
    ref_for_bert: List[str] = []
    cand_for_bert: List[str] = []

    for _, row in df.iterrows():
        t_payload = _parse_cell(row["target"])
        p_payload = _parse_cell(row["prediction"])
        gt_issues = parse_issues_payload(t_payload)
        pr_issues = parse_issues_payload(p_payload)

        yt = _safe_int_label(row["label"])
        yp = 1 if len(pr_issues) > 0 else 0
        y_true.append(yt)
        y_pred_bin.append(yp)

        miou, pairs = mean_greedy_iou_and_pairs(gt_issues, pr_issues)
        iou_row_means.append(miou)

        for g_issue, p_issue in pairs:
            g_txt = g_issue[2]
            p_txt = p_issue[2]
            if g_txt or p_txt:
                ref_for_bert.append(g_txt if g_txt else " ")
                cand_for_bert.append(p_txt if p_txt else " ")

    report: MetricsReport = {
        "n_rows": len(df),
        "binary": binary_classification_metrics(y_true, y_pred_bin),
        "line_range_iou": {
            "mean_iou_over_rows": (
                float(sum(iou_row_means) / len(iou_row_means)) if iou_row_means else 0.0
            ),
            "description": "Mean greedy IoU between GT and predicted line ranges; "
            "averaged per row over GT issues, then over rows.",
        },
    }

    if skip_bertscore:
        report["bertscore"] = {}
        report["bertscore_per_prediction"] = {}
        report["bertscore_skipped"] = "skipped (--skip-bertscore)"
        logger.info("BERTScore skipped by flag.")
        return report

    if comment_evaluator is None:
        comment_evaluator = CommentEvaluator()

    logger.info(
        "BERTScore: per model issue, max over human issues in the same row (argmax F1)..."
    )
    report["bertscore_per_prediction"] = compute_bertscore_per_prediction_max_over_refs(
        df, comment_evaluator
    )

    if ref_for_bert:
        logger.info(
            "BERTScore: %d greedy IoU-matched (GT-centric) comment pairs...",
            len(ref_for_bert),
        )
        bs = comment_evaluator.evaluate_to_dict(
            references=ref_for_bert,
            candidates=cand_for_bert,
        )
        report["bertscore"] = {
            "bertscore_precision": bs["mean_precision"],
            "bertscore_recall": bs["mean_recall"],
            "bertscore_f1": bs["mean_f1"],
            "matched_pairs": len(ref_for_bert),
            "description": "Mean P/R/F1 over pairs aligned by greedy line-range IoU (one pred per GT).",
        }
    else:
        report["bertscore"] = {}
        report["bertscore_greedy_skipped"] = "no non-empty greedy IoU-matched pairs"
        logger.info("Greedy IoU BERTScore skipped: no matched pairs.")
    return report


def evaluate_predictions_csv(
    csv_path: str,
    output_json_path: Optional[str] = None,
    skip_bertscore: bool = False,
) -> MetricsReport:
    """Load CSV, run :func:`evaluate_predictions_dataframe`, optionally save JSON."""
    logger.info("Loading predictions CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    report = evaluate_predictions_dataframe(df, skip_bertscore=skip_bertscore)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Wrote metrics JSON to %s", output_json_path)

    return report


def _log_report(report: MetricsReport) -> None:
    b = report.get("binary", {})
    logger.info("--- Binary (label vs any predicted issue) ---")
    logger.info("Accuracy:  %.4f", b.get("accuracy", 0.0))
    logger.info(
        "Precision (class 1): %.4f",
        b.get("precision_class_1_blocking", 0.0),
    )
    logger.info("Recall (class 1):    %.4f", b.get("recall_class_1_blocking", 0.0))
    logger.info("F1 (class 1):        %.4f", b.get("f1_class_1_blocking", 0.0))
    logger.info("Confusion [tn fp / fn tp]: %s", b.get("confusion_matrix", {}))

    iou = report.get("line_range_iou", {})
    logger.info("--- Line-range IoU (greedy GT match) ---")
    logger.info("Mean IoU: %.4f", iou.get("mean_iou_over_rows", 0.0))

    bsp = report.get("bertscore_per_prediction", {})
    if bsp:
        logger.info("--- BERTScore per model issue (max over humans in row) ---")
        logger.info(
            "n model issues: %s | mean F1: %.4f | median F1: %.4f",
            bsp.get("n_model_issues", 0),
            bsp.get("mean_f1", 0.0),
            bsp.get("median_f1", 0.0),
        )
        logger.info(
            "mean P/R: %.4f / %.4f | median P/R: %.4f / %.4f",
            bsp.get("mean_precision", 0.0),
            bsp.get("mean_recall", 0.0),
            bsp.get("median_precision", 0.0),
            bsp.get("median_recall", 0.0),
        )

    bs = report.get("bertscore", {})
    if bs:
        logger.info("--- BERTScore greedy IoU pairs (GT-centric) ---")
        logger.info(
            "BERTScore P/R/F1: %.4f / %.4f / %.4f",
            bs.get("bertscore_precision", 0.0),
            bs.get("bertscore_recall", 0.0),
            bs.get("bertscore_f1", 0.0),
        )
        logger.info("Pairs: %s", bs.get("matched_pairs", 0))
    elif report.get("bertscore_skipped"):
        logger.info("BERTScore: %s", report["bertscore_skipped"])
    elif report.get("bertscore_greedy_skipped"):
        logger.info("Greedy BERTScore: %s", report["bertscore_greedy_skipped"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate review predictions CSV (target vs prediction vs label).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to predictions CSV",
    )
    parser.add_argument(
        "-o",
        "--output-json",
        default=None,
        help="Write metrics to this JSON file",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        default=True,
        help="Skip BERTScore (faster, no GPU/model download)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    report = evaluate_predictions_csv(
        args.csv_path,
        output_json_path=args.output_json,
        skip_bertscore=args.skip_bertscore,
    )
    _log_report(report)


if __name__ == "__main__":
    main()
