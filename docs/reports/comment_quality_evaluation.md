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

## 6. Final Validation: Side-by-Side (SbS) Evaluation

While CodeBERTScore is used for fast iterations, our **Final Benchmark** to present to business stakeholders will utilize a **Side-by-Side (SbS) LLM-as-a-Judge** approach. We will evaluate our AI against human reviewers across a representative sample of **~100 Pull Requests**.

### 6.1 Evaluation Scope (PR-Level)
To ensure the benchmark is representative of real-world impact, we evaluate at the **Pull Request level**, not just the individual comment level.
* We extract the AI's predictions for *all changed files* in the PR.
* Even if the human reviewer did not leave a comment on a specific file, that file is still included in the SbS comparison to catch False Positives (AI noise) or False Negatives (Human oversights).

### 6.2 Interpretability Rubric (1-5 Scale)
To ensure the LLM Judge provides interpretable and reproducible results, it will score both the Human and the AI independently on a strict 1 to 5 scale:

*   **5 - Excellent:** Correctly identifies a critical bug or logical flaw and provides a clear, actionable fix. (Or, correctly remains silent when the code is flawless).
*   **4 - Helpful:** Technically correct and useful, but wording could be clearer or addresses a less critical issue (e.g., code style).
*   **3 - Neutral:** A trivial comment that neither harms nor significantly helps (e.g., nitpicking).
*   **2 - Poor:** Missed a real bug (False Negative), or flagged a non-existent issue causing cognitive noise (False Positive).
*   **1 - Harmful:** Severe hallucination, wrong localization, or provides advice that breaks the code.

### 6.3 Handling the "Human Silence" Case
A critical edge case is when a human did not comment, but the AI did (or vice versa). The Judge will use the rubric to score both:
*   **AI found a real bug that the human missed:** The Judge scores **AI = 5** (bug found) and **Human = 2** (bug missed).
*   **AI hallucinated a bug where the code was fine:** The Judge scores **Human = 5** (correctly remained silent) and **AI = 2** (false positive / noise).

### 6.4 Win Rate Calculation
The final business metric is the **AI Win Rate**. For every evaluated file in the PR:
*   **AI Win:** AI Score > Human Score
*   **Tie:** AI Score == Human Score
*   **AI Loss:** AI Score < Human Score

The final acceptance criteria will be determined by the aggregate Win/Tie percentage across the 100 PR validation set.
