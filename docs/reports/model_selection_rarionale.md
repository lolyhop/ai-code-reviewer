# Benchmark for evaluation and selection of the best LLM models

This document outlines which LLM models we are going to use in our solution, why we chose them, and how we are going to evaluate our results.

## 1. Benchmark for choosing the best models

Popular benchmarks don't directly evaluate how good the model is at reviewing pull requests and catching bugs. Therefore, we are going to use *SWE-bench verified* as a proxy. It tests AI systems' ability to solve GitHub issues. It should accurately approximate how well the model is at finding bugs since if it can fix them, it should also be able to spot them.

SWE-bench evaluation works as follows. Per task instance, an AI system is given the issue text. The AI system should then modify the codebase in order to resolve the described issues.

Because our compute resources are limited, we can only use models with fewer than 2B parameters in our solution.

## 2. Selecting best (small) reference models for our task

Once our model is ready, we will compare its performance with the following models (all can be found on HuggingFace):

| Model         | Parameters    | SWE-bench |
| ------------- | ------------- | --------  |
| SWE-Star-7B   | 7B            | 32.8%     |
| mini-coder-4b | 4B            | 26.8%     |
| mini-coder-1.7b | 1.7B        | 18.6%     |
| SWE-agent-LM-7B | 7B          | 15.2%     |
| Code_Review_Assistant_Model | 8B | ?      |

All of them perform relatively well for their size. The last model isn't evaluated on SWE-bench but has a similar idea to ours.

## 3. Selecting our base model
As the base model we plan to use [Qwen-3.5-2B-base](https://huggingface.co/Qwen/Qwen3.5-2B-Base). Reasons for this:

1) The base Qwen-3.5 version with 397B parameters performs well on the SWE-bench verified benchmark.
2) All of our reference models use older versions of Qwen, meaning it should perform well after fine-tuning.
3) It's very computationally efficient. The cost of generating 1M tokens with API is about 18 times less than for Gemini.
4) It's open-source.

## 4. Final evaluation

For offline evaluation we are going to compare our model with reference models on the following metrics:

*   **Precision (High Priority):** > 70%. We must minimize False Positives to avoid developer fatigue.
*   **Recall:** > 50% for critical categories.
*   **Semantic Similarity (Generated Comments):**
    *   **BERTScore:** Calculate similarity between generated comment and human ground truth.
    *   **LLM-Eval (Optional):** Use Qwen API to rate the helpfulness (1-5) of a sample of N generated comments.
    *   *Reference:* `docs/reports/5_evaluation.md` will contain the final metrics table.
