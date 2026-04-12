# Baseline Model Selection for Automated Pull Request Reviewer

This document explains how we select the baseline model for our project and how we plan to use it in our system.

Our goal is not to build the perfect model from the start. First, we need a small and practical baseline that we can run locally, evaluate on our dataset, and later fine-tune for our pull request review task.

## 1. Why we need a baseline model

Our system reviews Python pull requests and tries to detect blocking issues in changed files. For each changed file, the model should answer three questions:

1. Does this file contain a blocking issue?
2. Which changed lines are problematic?
3. What short review comment should be shown to the developer?

Before training our own model, we need a baseline model that can already do this task reasonably well in zero-shot or instruction-following mode. This baseline will help us:
- build the first inference pipeline;
- measure initial quality on an evaluation set;
- understand common model errors;
- compare the baseline with the fine-tuned version later.

In other words, the baseline gives us the first working version of the reviewer.

## 2. Model requirements

Our project is limited by available compute, so model selection is constrained not only by benchmark quality, but also by hardware.

We have access to a single **NVIDIA A100 with 80 GB VRAM**. This is enough for inference and for full fine-tuning of small models with longer context, but it still does not allow us to comfortably train larger LLMs without strong memory constraints.

When estimating whether a model can be trained on our hardware, we account for more than just model weights. During training, GPU memory is used for:
- model weights;
- gradients;
- optimizer states;
- activations saved for backpropagation;
- temporary CUDA buffers.

For full fine-tuning in mixed precision with AdamW, we estimate memory as **18 bytes per parameter**, plus activation memory. This comes from:

- **6 bytes per parameter** for model weights in mixed precision;
- **8 bytes per parameter** for AdamW optimizer states;
- **4 bytes per parameter** for gradients.

So the memory needed only for model states is:
- **1B parameters** -> about **18 GB**
- **1.5B parameters** -> about **27 GB**
- **2B parameters** -> about **36 GB**
- **4B parameters** -> about **72 GB**

Our GPU has **80 GB VRAM**, which means that even a **4B** model could fit for full fine-tuning by model states alone, and a **2B** model leaves substantial headroom for activations.

### Activation memory and context length

After subtracting model states from GPU memory, the remainder is available for activation memory, which scales linearly with sequence length (assuming gradient checkpointing and Flash Attention). For **Qwen3-1.7B** on an **A100 80 GB**:

- Model states: **1.7B × 18 bytes ≈ 30.6 GB**
- CUDA overhead (buffers, fragmentation): **~2 GB**
- Remaining for activations: **80 − 30.6 − 2 ≈ 47.4 GB**

For comparison, an A100 40 GB would leave only ~7 GB for activations, enough for roughly 4K tokens. With 47.4 GB (about **6.7× more**), we can scale proportionally:

| Context length | Activation estimate | Fits on A100 80 GB? |
|---:|---:|---|
| 4,096 | ~7 GB | Yes (comfortable) |
| 8,192 | ~14 GB | Yes (comfortable) |
| 16,384 | ~28 GB | Yes |
| 24,576 | ~42 GB | Tight |
| 32,768 | ~56 GB | No |

During the Data Preparation stage, we estimated the target input length for our system. A single review prompt includes the changed file context, pull request metadata, repository-level metadata, up to **3 imported-code snippets**, up to **3 usage-site snippets**, and the task instruction. Based on this prompt structure and the activation budget above, we use **16,384 tokens** as the target context length for training and inference. This covers approximately **75–80%** of the samples in our dataset after applying context compression heuristics.

So our baseline model must satisfy four requirements:

- it must fit on a single **A100 80 GB** GPU;
- it must support inference on inputs up to **16,384 tokens**;
- it must be suitable for full fine-tuning or parameter-efficient fine-tuning (PEFT);
- it must have strong **code understanding ability**.

These constraints narrow our search to compact code-oriented models in the **1.5B–2B range**.

## 3. Candidate models

There is no standard benchmark that directly measures how well an LLM can review Python pull requests. Because of this, we use **LiveCodeBench** as the main benchmark for model pre-selection ([link to the benchmark](https://livecodebench.github.io)). Our assumption is simple: a model that is good at solving fresh coding tasks is more likely to be good at reading code diffs, reasoning about bugs, and writing useful review comments. LiveCodeBench is a good fit for this purpose because it is designed as a **contamination-free coding benchmark** and is built from continuously collected recent programming problems rather than old static test sets. In LiveCodeBench, the main metric is **Pass@1**, which measures how often the model solves the task correctly on its first attempt. For this metric, **higher is better**. 

We do not treat LiveCodeBench as a direct replacement for our final project evaluation. We use it only to choose a strong baseline family before running our own experiments on the review dataset. For our project, this is a practical proxy: PR review is not the same as competitive programming, but both tasks require code understanding, bug reasoning, and precise generation under constraints.

We selected five open models that are realistic for our setup and have usable evidence on LiveCodeBench. We intentionally include both small and medium-size models, because we want to balance **benchmark quality**, **hardware fit**, and **training feasibility**. The table below summarizes the candidates.

| Model | Params | Type | Context length | LiveCodeBench Pass@1 ↑ |
|---|---:|---|---:|---:|
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | general | 32,768 | 33.2 |
| [OpenCoder-1.5B-Instruct](https://huggingface.co/infly/OpenCoder-1.5B-Instruct) | 1.5B | code instruct | 4,096 | 12.8 |
| [DeepSeek-Coder-1.3B-Instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) | 1.3B | code instruct | 16,384 | 5.1 |
| [Yi-Coder-1.5B-Chat](https://huggingface.co/01-ai/Yi-Coder-1.5B-Chat) | 1.5B | code chat | 128,000 | 4.8 |
| [CodeLlama-7B-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) | 7B | code instruct | 16,384 | 7.1 |

The score for **Qwen3-1.7B** comes from the official Qwen3 benchmark results. The scores for **DeepSeek-Coder-1.3B-Instruct**, **Yi-Coder-1.5B-Chat**, and **CodeLlama-7B-Instruct** come from the **Qwen2.5-Coder technical report**, where these models are compared on LiveCodeBench together with other coding benchmarks. The score for **OpenCoder-1.5B-Instruct** comes from its Hugging Face model card, which reports **12.8** on LiveCodeBench. Since these values come from different sources, we do not treat them as perfectly comparable, but they are still useful for selecting a practical baseline family.

Among the models in the small-model range, **Qwen3-1.7B** stands out as the strongest candidate for our project. It has the highest reported LiveCodeBench score among the models in our target size range, supports a **32,768-token** context window, and still fits our hardware constraints. This is important for our system because our prompts include not only the changed file, but also pull request metadata, repository metadata, imported-code snippets, usage-site snippets, and task instructions.

Based on this comparison, we choose **Qwen3-1.7B** as the main baseline model for the first inference pipeline. It gives us the best balance of **small size**, **strong coding performance**, and **long context support**. We keep **OpenCoder-1.5B-Instruct** as the main alternative baseline because of its strong reported LiveCodeBench result, and we keep **DeepSeek-Coder-1.3B-Instruct** and **Yi-Coder-1.5B-Chat** as secondary comparison models. We do not prioritize **CodeLlama-7B-Instruct**, because it is substantially larger and less convenient for lightweight iterative experiments and fine-tuning in our setup.

## 4. How the baseline will be used

We will first use the selected baseline model in a zero-shot inference pipeline. For each row in the evaluation dataset, the system will build a review prompt, run the model, and parse the answer into a standard structured format with two fields: predicted line range and review comment.

Then we will evaluate the baseline on our validation set using detection metrics, localization quality, and comment similarity. These results will serve as the main reference point before supervised fine-tuning.

After that, we will fine-tune the same model family on the cleaned training set and compare the fine-tuned model against the baseline under the same evaluation setup.

The rest of the implementation details will be handled in code. This includes prompt design, inference scripts, output parsing, evaluation logic, data cleaning, and fine-tuning experiments. This document only fixes the baseline model choice and explains why this choice is reasonable for our project.