# Methodology: Generated Comment Quality Evaluation

## 1. Executive Summary
Evaluating AI-generated code review comments is more complex than bug detection. While bug detection is a simple "Yes/No" (binary) task, a review comment can be written in many different ways. Traditional metrics that look for exact word matches (like BLEU or ROUGE) often fail because they don't understand that two different sentences can have the same meaning.

For this project, we prioritize **semantic similarity** - measuring if the AI's advice means the same thing as the human's advice.

## 2. BERTScore Explained
BERTScore is our primary metric for comparing a model-generated comment against a human reference comment.

### 2.1 How it Works
Instead of matching exact words, BERTScore uses a Transformer model to turn words into "embeddings" (mathematical vectors of meaning).
1. **Represent:** It turns every word in both comments into a vector.
2. **Match:** It finds the best "meaning match" for every word in the AI comment against the human comment.
3. **Score:**
    *   **Precision:** Did the AI say anything that was NOT in the human's notes? (Focuses on truthfulness).
    *   **Recall:** Did the AI catch everything the human mentioned? (Focuses on completeness).
    *   **F1 Score:** The overall balance. **This is our main KPI.**

### 2.2 Why use it?
It is robust to **paraphrasing**. If a human says *"Unused variable"* and the AI says *"This variable is never used,"* BERTScore will give a high score, whereas older metrics would give a low score.

## 3. The "Code" Evolution: CodeBERTScore
Standard BERTScore is trained on books and Wikipedia. For this project, we use **CodeBERTScore** (BERTScore using the `microsoft/codebert-base` model).
*   **Why:** It understands Python syntax. It knows that `x += 1` and `x = x + 1` are the same thing, while a regular language model might get confused by the different symbols.

## 4. Comparison of Evaluation Approaches

| Approach | Metric Type | Reliability | Speed | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **ROUGE-L** | Word Overlap | Low (misses meaning) | Very Fast | Zero |
| **CodeBERTScore** | Semantic | **High (Best for AI)** | Medium | Low (Runs on CPU/GPU) |
| **LLM-as-a-Judge** | Reasoning | Highest | Slow | High (API costs/Time) |

## 5. Proposed Methodology (3-Tier Evaluation)

To ensure our PR Reviewer is high quality, we will use three layers of testing:

1.  **Tier 1: Automated Scoring (Primary).** We will run **CodeBERTScore (F1)** on every test. Our goal is to see this score increase as we fine-tune our model.
2.  **Tier 2: Hallucination Check (Logic).** A simple script will check if the AI mentions variable names or line numbers that do not exist in the code diff. If it does, the comment is rejected.
3.  **Tier 3: Expert Validation (Human).** Once a week, we will manually review 20 comments to ensure they are actually helpful and have a professional tone.
