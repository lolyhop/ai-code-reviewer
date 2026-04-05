# Methodology: Offline Evaluation of Generated Review Comments

## 1. The Evaluation Problem
Evaluating AI-generated code review comments is fundamentally different from binary bug classification. For code reviews, a generated comment can be perfectly valid even if it shares zero words with the reference (e.g., *"Remove this"* vs. *"This is unused"*).

Standard exact-match metrics (like BLEU or ROUGE) rely on lexical overlap. They heavily penalize paraphrasing and fail to capture semantic equivalence. Therefore, we must define an evaluation methodology that relies on **semantic similarity**.

## 2. Candidate Evaluation Approaches

Before selecting our primary metric, we analyzed three fundamentally different approaches for evaluating `<prediction, target>` pairs.

### 2.1 Lexical Overlap: ROUGE-L
ROUGE-L measures the Longest Common Subsequence (LCS) between the prediction and the target. [Link to Paper](https://aclanthology.org/W04-1013/).
*   **Mechanism:** Counts words appearing in the same relative order.
*   **Pros:** Computationally trivial, near-zero latency.
*   **Cons:** Cannot detect synonyms. A syntactically broken sentence with overlapping words will score higher than a perfect paraphrase. **Rejected for semantic evaluation.**

### 2.2 Reasoning-Based: LLM-as-a-Judge
Using a large language model (e.g., GPT-4 or Qwen-72B) prompted with a grading rubric to evaluate the prediction. [Link to Paper](https://arxiv.org/abs/2303.16634).
*   **Mechanism:** The LLM receives the diff and both comments, outputting a score based on semantic helpfulness.
*   **Pros:** Highly correlated with human judgment; provides reasoning for interpretability.
*   **Cons:**
    *   **Data Privacy:** We are building an *On-Premise* solution for enterprise clients with sensitive code. Sending data to 3rd-party APIs (like OpenAI) is strictly prohibited.
    *   **Hardware Limits:** To act as a reliable "Judge", an LLM needs significant reasoning capabilities (typically >7B parameters). Our local inference constraints limit us to smaller models (~2B parameters) for fast tasks. Hosting a heavy >7B model locally takes **2000+ ms per evaluation** and consumes massive GPU resources.
*   **Decision:** **Rejected for offline iterative loops** due to hardware and speed constraints, but **accepted for the final representative evaluation** (where slow, high-quality local inference is acceptable).

### 2.3 Contextual Embeddings: BERTScore
BERTScore computes token-level semantic similarity using pre-trained language models. [Link to Paper](https://arxiv.org/abs/1904.09675).
*   **Mechanism:** Maps tokens to dense vector spaces and uses greedy cosine similarity matching.
*   **Pros:** Robust to paraphrasing; captures deep semantic meaning; runs locally on standard GPUs.
*   **Decision:** Selected as the foundational methodology for our offline evaluation.

## 3. BERTScore Explained

To understand why BERTScore is effective, we must look at its mathematical implementation. BERTScore calculates similarity not at the sentence level, but by creating a bipartite graph of token embeddings.

![BERTScore Matching Example](https://github.com/Tiiiger/bert_score/blob/master/bert_score.png?raw=true)

### 3.1 Tokenization and Embedding
Given a reference human comment $x$ and a generated candidate comment $\hat{x}$:
1. Both sentences are passed through a Transformer tokenizer.
2. The encoder generates contextual embeddings: $\mathbf{x}_i$ for each token in the reference, and $\mathbf{\hat{x}}_j$ for each token in the candidate.

### 3.2 The Cosine Similarity Matrix

Before computing similarities, all token embeddings are **L2-normalized** (unit vectors). This reduces the dot product to cosine similarity:

$$
\text{sim}(\mathbf{x}_i, \mathbf{\hat{x}}_j) = \frac{\mathbf{x}_i^\top \mathbf{\hat{x}}_j}{\|\mathbf{x}_i\| \|\mathbf{\hat{x}}_j\|}
$$

We compute this for every possible token pair, producing a similarity matrix
that feeds into the greedy matching step below.

### 3.3 Greedy Matching (Precision, Recall, F1)
Instead of matching words exactly, BERTScore uses greedy matching in the embedding space:

*   **Precision ($P_{BERT}$):** For every token in the *generated* comment, find the most semantically similar token in the *reference*. It penalizes the model if it hallucinates concepts not present in the human reference.

$$
    P_{BERT} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_j \in \hat{x}} \max_{x_i \in x} \mathbf{x}_i^\top \mathbf{\hat{x}}_j
$$

*   **Recall ($R_{BERT}$):** For every token in the *reference*, find the most similar token in the *generated* comment. It penalizes the model if it fails to address a critical point made by the human.

$$
    R_{BERT} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} \mathbf{x}_i^\top \mathbf{\hat{x}}_j
$$

*   **F1 Score ($F_{BERT}$):** The harmonic mean of P and R. This serves as our primary evaluation metric.

$$
    F1_{BERT} = \frac{2 \times P_{BERT} \times R_{BERT}}{P_{BERT} + R_{BERT}}
$$

#### Worked Example

Given:
- Reference (*x*): `["unused", "variable", "remove"]`
- Candidate (*x̂*): `["never", "used", "delete"]`

**Step 1 — Build the cosine similarity matrix:**

|              | `unused` | `variable` | `remove` |
| :----------- | :------: | :--------: | :------: |
| **`never`**  |   0.84   |    0.12    |   0.20   |
| **`used`**   |   0.79   |    0.55    |   0.15   |
| **`delete`** |   0.10   |    0.08    |   0.91   |

**Step 2 — Greedy match each candidate token to its best reference token:**
- `"never"` → max(0.84, 0.12, 0.20) = **0.84**
- `"used"` → max(0.79, 0.55, 0.15) = **0.79**
- `"delete"` → max(0.10, 0.08, 0.91) = **0.91**

**Step 3 — Average to get Precision:**
$$P_{BERT} = \frac{0.84 + 0.79 + 0.91}{3} = 0.85$$

Recall is computed symmetrically (each *reference* token matches its best candidate token). F1 is the harmonic mean of both.

## 4. From BERTScore to CodeBERTScore

### 4.1 Can Any BERT Model Be Used?

Technically yes — BERTScore accepts any HuggingFace encoder as its backbone.
However, the choice of backbone significantly affects scores because tokenizers
differ: BERT and RoBERTa use WordPiece, CodeBERT uses BPE, T5 uses SentencePiece.
Scores computed with different backbones are **not comparable to each other**, so
the backbone must stay fixed for the entire evaluation run.

### 4.2 The Tokenizer Problem
Standard BERT models (like `bert-base-uncased`) use WordPiece tokenizers trained on Wikipedia. When they encounter code snippets within a review comment (e.g., `x += 1`), they shatter the syntax into meaningless sub-tokens (`x`, `+`, `=`, `1`), destroying the semantic representation.

### 4.3 The Solution: `microsoft/codebert-base`
To resolve this, we parameterize our BERTScore implementation to use CodeBERT. [Link to Paper](https://arxiv.org/abs/2002.08155)
*   **Code-Aware Vocabulary:** CodeBERT was pre-trained on CodeSearchNet. Its tokenizer (BPE) preserves programming identifiers, brackets, and operators.
*   **Domain Context:** The embeddings understand that `list.append(x)` and `list.insert(len(list), x)` are semantically equivalent in Python.

*(Note: This aligns with the principle behind [CodeBERTScore (Ren et al., 2023)](https://arxiv.org/abs/2302.05527), which demonstrates that code-pretrained backbones yield higher correlation with human judgment on code-related tasks than standard BERTScore. The original paper targets NL→code generation; our adaptation applies the same backbone-substitution principle to NL→NL review comment evaluation.)*

## 5. Methodological Summary & Benchmark

The following data was gathered via a local benchmark script (100-runs average) on a Python idiom example:

- *Ref: "Use increment operator instead of x = x + 1."*
- *Pred: "x += 1 is preferred over x = x + 1."*

| Metric | Mechanism | Tokenizer | Avg Latency | Score (Semantic Match) |
| :--- | :--- | :--- | :--- | :--- |
| **ROUGE-L** | Lexical (LCS) | N/A | **~0.2 ms** | 0.3750 (Fails) |
| **BERTScore** | Semantic | WordPiece | ~140 ms | 0.7038 (Partial) |
| **CodeBERTScore**| Semantic | **BPE (Code)** | **~120 ms** | **0.9194 (Optimal)** |
| **LLM-Judge\*** | Reasoning | N/A | ~2000+ ms | ~0.95 (Gold Standard) |

*\* Latency is referenced from local inference speeds of >7B parameter models (e.g., Llama-3-8B) on standard enterprise GPUs (NVIDIA T4 16GB), which average 20-30 tokens/sec. Processing the prompt context and generating Chain-of-Thought reasoning inherently requires >2 seconds per sample.*

### 5.1 Iterative Evaluation Conclusion
For all fast offline evaluation and model tuning, we will utilize **CodeBERTScore (Mean F1)**. It provides the necessary semantic depth to recognize Python idioms while maintaining the computational efficiency required for automated training loops, adhering to our On-Premise hardware limitations.

## 6. Final Validation: LLM-as-a-Judge Evaluation Framework

While CodeBERTScore is used for fast iterative evaluation and model tuning, our **Final Benchmark** to present to business stakeholders will utilize an **LLM-as-a-Judge** approach. We will evaluate our AI against human reviewers across a representative sample of **~100 Pull Requests**, providing a rigorous, interpretable assessment of real-world competitive performance.

### 6.1 Judge Selection and Reasoning

#### 6.1.1 Candidate Judge Models
Based on recent benchmarking literature ([Judge's Verdict](https://arxiv.org/html/2510.09738v1), [CodeJudgeBench](https://arxiv.org/pdf/2507.10535), [Judging the Judges: A Systematic Investigation of Position Bias](https://arxiv.org/html/2406.07791v5)), we evaluated the following candidates:

**Top-Tier Proprietary Models:**
- GPT-4.5, GPT-4o, Claude-3.5-Sonnet: Excellent correlation with human judgment (κ > 0.80), but proprietary and API-based only.

**Open-Source Candidates:**
- Llama-3.1-70B, Qwen3-72B: Strong performance (κ > 0.78), available for local deployment.
- **DeepSeek-R1-Distill-Qwen-32B/70B:** Specialized for reasoning and coding tasks, with strong performance on CodeJudgeBench (~73% average on coding tasks).

#### 6.1.2 Selection Rationale: DeepSeek-R1-Distill-Qwen

We selected **DeepSeek-R1-Distill-Qwen-32B** (with 70B as fallback) for the following reasons:

1. **On-Premise Compatibility:** Our project operates under **strict security and privacy policies** prohibiting the use of proprietary APIs (OpenAI, Anthropic, Google). DeepSeek models are fully open-source and can be deployed locally within enterprise infrastructure.

2. **Cost Efficiency:** DeepSeek-R1-Distill-32B provides a significant performance boost compared to base models (~72% coding accuracy) while remaining deployable on standard enterprise GPUs (single NVIDIA H100 or equivalent).

3. **Code-Specific Reasoning:** DeepSeek-R1 incorporates chain-of-thought reasoning optimized for coding tasks (as evidenced by CodeJudgeBench results), which is critical for evaluating code review comment quality in context.

4. **Bias Awareness:** DeepSeek-R1-Distill models have demonstrated lower position bias compared to earlier-generation models (as shown in [Comparing Developer and LLM Biases in Code Evaluation](https://arxiv.org/html/2603.24586v1)), allowing for more robust multi-pass evaluation.

5. **Reduced Hallucination:** The distilled versions are optimized for factual accuracy, reducing the risk of catastrophic errors (score 1, "Harmful") when evaluating code.

**Trade-off Acceptance:** While top-tier proprietary models (GPT-4.5, Claude-3.5) show slightly higher human agreement (κ ≈ 0.81 vs. κ ≈ 0.78 for open-source), the alignment gap to human judgment remains measurable (~5-12% score deviation). This trade-off is acceptable in exchange for full on-premise deployment and auditability.

### 6.2 Comprehensive Evaluation Methodology

#### 6.2.1 Input Context and Data Flow

Unlike our fast iterative evaluation (CodeBERTScore), the Judge receives **enriched context** to minimize hallucination and improve reasoning accuracy:

**Inputs per Pull Request:**
1. **PR Metadata** (summary context):
   - PR title, description, and intent
   - List of changed files and their impact indicators

2. **Full File Context** (per analyzed file):
   - Complete patched file content (not truncated, to preserve full context)
   - File path and language hints
   - Diff annotations marking added/deleted/modified lines

3. **Dependency Context** (when applicable):
   - Imported module definitions (up to 3 modules per file)
   - Function/class definitions from imported modules that are actively used in the changed code
   - Repository configuration files (pyproject.toml, requirements.txt) in summarized form

4. **Related Changes** (from the same PR):
   - For files modified in the same PR, related file content is included to detect cross-file impacts
   - This prevents the Judge from missing issues that span multiple files

5. **Review Pair**:
   - **Review A (Human):** Set of comments from the original human reviewer, with line annotations
   - **Review B (AI):** Set of predicted comments from our AI system, with predicted line numbers

#### 6.2.2 Judge Role and Scoring Methodology

**Judge Role:** The Judge acts as a **Senior Technical Lead** who must evaluate whether each review comment (human or AI) would be valuable in a real code review setting.

**Scoring Scale (1–5):**

- **5 – Excellent:** Correctly identifies a critical bug, security issue, or logical flaw with clear, actionable guidance. Sets best practices or prevents production incidents. *(Or correctly remains silent if the code has no issues.)*
  - Examples: "Null pointer dereference risk," "SQL injection vulnerability," "Race condition in async code."

- **4 – Helpful:** Technically sound and useful, but addresses a non-critical issue or could be phrased more clearly. Improves code quality but is not merge-blocking.
  - Examples: "Consider using list comprehension for clarity," "Variable name is unclear."

- **3 – Neutral:** Trivial comment with marginal value. Neither significantly harms nor helps.
  - Examples: Nitpicking on style, redundant formatting suggestions.

- **2 – Poor:** False Negative (missed a real bug) or False Positive (flagged a non-issue, creating noise).
  - Examples: Incorrectly claiming a variable is unused when it is used elsewhere, or suggesting a refactor that changes functionality.

- **1 – Harmful:** Severe hallucination, incorrect code localization, or suggestions that would break the code if implemented.
  - Examples: Recommending deletion of critical code, suggesting syntax-invalid replacements.

**Independent Scoring:** The Judge scores the Human review and the AI review **independently**. This ensures that if the human made errors or one side is completely absent, those facts are captured in the scores.

#### 6.2.3 Handling Review Cardinality (N:M Matching)

A key challenge in comparative evaluation is that humans and AI may produce different numbers of comments, and they may not directly correspond.

**Strategy:** The Judge is instructed to **holistically evaluate file-level review coverage**, not perform one-to-one comment matching. Specifically:

- The Judge assesses: *"Did this reviewer catch the important issues in this file? Did they avoid creating noise?"*
- The Judge is made aware of both review sets simultaneously, allowing implicit understanding of overlaps and gaps.
- If a comment occurs in only one set, the Judge's independent score captures whether it was a missed issue (**Human misses AI's finding** → Human score lower) or a hallucination (**AI invents AI's finding** → AI score lower).

#### 6.2.4 Pull Request-Level Aggregation

Evaluation proceeds at the PR level across all changed files:

1. Each changed Python file in the PR is evaluated.
2. A file-level "Win" is determined as: *AI Score > Human Score* (across all comments for that file).
3. A file-level "Tie" is: *AI Score == Human Score*.
4. A file-level "Loss" is: *AI Score < Human Score*.

**PR-Level Win Rate:**
$$\text{Win Rate} = \frac{\text{\# Files with AI Win} + 0.5 \times \text{\# Files with Tie}}{\text{Total \# Files}}$$

This metric expresses: *"In what fraction of files does the AI provide review quality comparable to or better than the human?"*

### 6.3 Prompt Template


The Judge receives a structured prompt designed for holistic, fair, and unbiased comparison of two sets of review comments for a single file. The prompt is designed for N:M review comparison with explicit rules to distinguish merge-blocking semantic issues from stylistic concerns. This section documents the production-grade prompt implemented in `notebooks/llm_judge.ipynb`.

#### 6.3.1 System Prompt (Production-Grade)

The actual prompt enforces strict evaluation rules distinguishing merge-blocking semantic issues from stylistic noise:

```
You are a strict Senior Staff Software Engineer acting as a code review judge for a Python codebase.
Your task is to objectively evaluate two sets of review comments—Review A and Review B—on a single Python file.

### ARCHITECTURAL CONSTRAINTS (CRITICAL)
This system is designed ONLY to catch **merge-blocking semantic issues** (e.g., logical correctness, security vulnerabilities, thread-safety, performance regressions, or severe maintainability flaws).
**Style, formatting, and minor naming conventions are OUT OF SCOPE** (they are handled by CI linters). Comments focusing purely on style should be treated as "Noise".

### EVALUATION PROTOCOL (STRICT)
1. **Hallucination Check (Score 1):** If a review mentions variables, loops, or logic that DO NOT EXIST in the provided FILE CONTENT, you MUST score it a 1 (Harmful).

2. **The "Noise" Penalty (Score 2 or 3):** If a review ignores critical bugs to focus purely on PEP8 formatting, docstrings, or minor naming, it is providing Noise.

3. **The "Silence" Evaluation:** If a review is empty:
   - If FILE CONTENT contains a blocking bug, the empty review missed it. Score = 1 or 2.
   - If FILE CONTENT is free of blocking bugs (even if style is bad), an empty review is correct. Score = 5.

4. **Outcome Selection:**
   - If Score A > Score B: "A Win"
   - If Score B > Score A: "B Win"
   - If Score A == Score B: "Tie"
```

#### 6.3.2 Review Rubric (Batch/N:M Comparison)
| Score | Description |
|-------|-------------|
| 5 | **Excellent:** Review covers all critical and important issues, provides clear and actionable feedback, and avoids noise or hallucinations. No significant issues missed. |
| 4 | **Strong:** Review addresses most important issues, feedback is mostly clear and actionable, but may miss a minor point or include a minor nitpick. |
| 3 | **Adequate:** Review covers some relevant issues but misses at least one important point, or includes some unnecessary comments. Value is mixed. |
| 2 | **Weak:** Review misses multiple important issues, or contains several incorrect, irrelevant, or noisy comments. May cause confusion or extra work. |
| 1 | **Harmful:** Review is misleading, mostly hallucinated, or would cause harm if followed (e.g., suggests breaking changes, or misses all critical issues). |

#### 6.3.3 Input Section (Per-File)
```
FILE INFORMATION:
- Path: {file_path}
- Language: Python
- Lines Changed: {line_range}

FILE CONTENT:
{full_file_content}

CONTEXT:
{PR_metadata}
{imported_module_snippets}
{related_file_content}

REVIEW A:
{review_a_comments_with_lines}

REVIEW B:
{review_b_comments_with_lines}
```

#### 6.3.4 Output Structure
```json
{
   "review_a_score": <1-5>,
   "review_a_reasoning": "...",
   "review_b_score": <1-5>,
   "review_b_reasoning": "...",
   "outcome": "A Win/B Win/Tie"
}
```

**Note:** The notebook stores these 5 fields for each bidirectional pass in `raw_judge_outputs.json` along with metadata (file_id, timestamps). The raw model output is preserved in `raw_deepseek_output` for debugging. This structure enables both summarized scoring and detailed post-hoc analysis for position-bias detection.

### 6.4 Bias Mitigation Strategy

#### 6.4.1 Position Bias Problem

Recent research ([Judging the Judges: A Systematic Investigation of Position Bias](https://arxiv.org/html/2406.07791v5), [Comparing Developer and LLM Biases in Code Evaluation](https://arxiv.org/html/2603.24586v1)) demonstrates that LLM judges exhibit **strong position bias**—a consistent preference for the first-presented option (typically 10–45% accuracy gap between position-consistent and overall accuracy). This bias compromises the validity of comparative evaluations.

#### 6.4.2 Bidirectional Scoring (Position Swap Mitigation)

We adopt a **bidirectional evaluation protocol** to control for position bias:

1. **First Pass:** Provide Review A (Human) in Position 1, Review B (AI) in Position 2.
   - Record scores: Human₁, AI₁.

2. **Second Pass:** Present the same reviews with **swapped positions**: Review B (AI) in Position 1, Review A (Human) in Position 2.
   - Record scores: Human₂, AI₂.

3. **Consistency Check:**
   - If Human₁ > AI₁ AND Human₂ > AI₂: **Consistent Human Win** (clean result).
   - If AI₁ > Human₁ AND AI₂ > Human₂: **Consistent AI Win** (clean result).
   - If the Judge flips its preference based on position: Result is marked **Tie** (inconclusive; position bias detected).

#### 6.4.3 Data Synthesis

For the final Win Rate calculation, only **consistent results** (no position-flip) are counted as definitive wins/losses. Position-inconsistent comparisons are treated as ties, which conservatively treats the AI as achieving parity rather than claiming spurious superiority.

#### 6.4.4 Related Debiasing Techniques

While position swapping is our primary mitigation, emerging research ([CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias](https://aclanthology.org/2025.acl-long.808.pdf)) suggests complementary techniques:

- **Debiasing Instructions:** Explicit guidance to avoid position bias (with limited effectiveness; aids but does not eliminate bias).
- **Contextual Calibration:** Applying affine transformations to model outputs to neutralize learned biases.

These techniques are marked as **future enhancements** if preliminary 1-pass results show unacceptable inconsistency.

### 6.5 Judge Calibration and Validation

Before trusting the Judge on the full 100 PR evaluation set, we perform a **calibration step** using the implementation in `notebooks/llm_judge.ipynb`:

1. **Manual Sample Set:** Annotate 10-15 representative `<file, Human Review, AI Review>` triplets by hand, assigning ground-truth scores without consulting the Judge.

2. **Judge Inference:** Run the Judge on the same 10-15 samples using the bidirectional evaluation protocol (Section 6.4.2). The notebook implements the complete workflow: data loading, model initialization, bidirectional inference with position swapping, and metrics computation.

3. **Agreement Metric:** Compute Cohen's κ (inter-rater reliability) between manual annotations and mean Judge scores (averaged over both forward passes).

4. **Acceptance Threshold:** If κ ≥ 0.70 (indicating "substantial agreement"), proceed to full evaluation. If κ < 0.70, refine the prompt or recalibrate the Judge threshold.

5. **Calibration Feedback:** Share results with the development team to enable iterative prompt refinement before the final benchmark.

**Hardware and Timeline Note:** The calibration notebook uses DeepSeek-R1-Distill-Qwen-32B, which requires significant GPU resources. Each sample undergoes **bidirectional inference** (2 forward passes to detect position bias), and each pass requires inference with ~500 output tokens for full reasoning. Wall-clock time per sample set (10-15 samples) depends on hardware:
- **GPU-accelerated environments (Colab/Kaggle T4/A100):** 50-75 minutes per calibration set
- **Enterprise GPUs (NVIDIA H100):** 3-5 minutes per calibration set
- **CPU-only execution:** Not recommended (hours to days per sample set)

Plan for **1-2 hours** in Kaggle/Colab with GPU runtime for complete calibration workflow including model loading, inference, and metrics computation.

### 6.6 Final Business Metric: AI Win Rate

The ultimate metric presented to business stakeholders is the **Aggregate Win Rate** across all ~100 evaluated PRs:

$$\text{AI Win Rate} = \frac{\sum_{\text{all files}} (\text{AI Score} - \text{Human Score}) > 0}{\text{Total \# Files Evaluated}}$$

**Interpretation:**
- **75% Win Rate** = *"In 75% of files, our AI provided review quality comparable to or better than Senior developers."*
- **60% Win Rate** = *"AI and humans are broadly equivalent; AI excels in specific domains."*
- **<50% Win Rate** = *"AI requires further refinement; human reviewers still significantly outperform."*

This metric directly maps to business value: higher Win Rates translate to reviewers spending less time on routine checks and focusing on complex architectural issues.
