# Automated Pull Request Reviewer — Demo

Streamlit demo for the [Automated Pull Request Reviewer](../README.md) project.
Pastes a GitHub PR URL, fetches the PR, builds the same review context used by
the project pipeline, runs **local** Qwen3 inference, and renders a
GitHub-like review.

## Quick start

```bash
# 1. Install demo + local-inference dependencies
pip install -e ".[finetune,demo]"

# 2. (recommended) export a GitHub token
export GITHUB_TOKEN=ghp_your_token_here

# 3. Run the demo
streamlit run demo/app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Startup

The local reviewer model is loaded **eagerly on app startup**, before the
PR URL input is rendered. Subsequent reruns reuse the cached resource via
`streamlit.cache_resource`. If model loading fails the URL input is hidden
and an error with install hints is shown instead.

## Backends

The demo supports two inference backends, selected via `APR_BACKEND`:

| Backend | When to use | Setup |
|---|---|---|
| `transformers` (default) | NVIDIA GPU, full HF parity with the rest of the project | `pip install -e ".[finetune,demo]"` |
| `ollama` | **Apple Silicon (M1/M2/M3)**, or any host without a CUDA GPU | `brew install ollama && ollama pull qwen2.5-coder:1.5b` |

Both backends use the **same `ReviewPipeline`** to build prompts and the
same JSON parser for outputs — only the generator changes.

### Why Ollama on Apple Silicon

`transformers` + MPS (the only PyTorch GPU backend on Mac) is much slower
than `llama.cpp` via Metal: bf16 ops fall back to CPU, no FlashAttention,
and several Qwen kernels are not fused on MPS. Ollama wraps llama.cpp,
runs natively on Metal, and is typically **5-10x faster** on the same
M-series chip. Generation also uses Ollama's `format: "json"` mode, which
forces valid JSON output and improves parse success on small models.

### Quick start with Ollama

```bash
brew install ollama
ollama serve &                    # start the local server on :11434
ollama pull qwen2.5-coder:1.5b    # ~1 GB GGUF download

APR_BACKEND=ollama \
APR_OLLAMA_MODEL=qwen2.5-coder:1.5b \
streamlit run demo/app.py
```

The startup spinner does both a health check (`/api/tags`) and a 1-token
warm-up so the first real `Analyze PR` click is not stuck on cold-load.

## Environment variables

### Common

| Variable | Default | Notes |
|---|---|---|
| `GITHUB_TOKEN` | _unset_ | Needed for private PRs and for >60 req/h on public PRs. |
| `APR_BACKEND` | `transformers` | One of `transformers`, `ollama`. |
| `APR_MAX_NEW_TOKENS` | `512` | Generation budget per file (forwarded to both backends). |

### Transformers backend (`APR_BACKEND=transformers`)

| Variable | Default | Notes |
|---|---|---|
| `APR_MODEL_NAME` | `Qwen/Qwen3-1.7B` | HF id or local path (see `docs/design/baseline_model.md`). |
| `APR_DEVICE` | `auto` | Forwarded to `device_map` in `ModelConfig`. |
| `APR_TORCH_DTYPE` | `bfloat16` | One of `float16`, `bfloat16`, `float32`. |
| `APR_MAX_INPUT_TOKENS` | `16384` | Truncation length for the prompt. |
| `APR_LOAD_IN_4BIT` | `false` | Set to `1`/`true` for 4-bit quantized loading (requires `bitsandbytes`). |

### Ollama backend (`APR_BACKEND=ollama`)

| Variable | Default | Notes |
|---|---|---|
| `APR_OLLAMA_MODEL` | `qwen2.5-coder:1.5b` | Tag must already be pulled (`ollama pull <tag>`). |
| `APR_OLLAMA_URL` | `http://localhost:11434` | Override only if you run Ollama on a different host/port. |

Inference is fully local in both backends; no proprietary code or generated
content is sent to external APIs.

## How it reuses the project pipeline

`demo/adapters.py` is intentionally thin and forwards to existing modules:

| Step | Existing module |
|---|---|
| Apply unified diff to base file -> annotated `patched_content` | `ai_code_reviewer.dataset.patches.compute_patched_content` |
| `ReviewSample` data class | `ai_code_reviewer.models.schema.ReviewSample` |
| Heuristic dependency retriever | `ai_code_reviewer.models.retriever.Retriever` (via the pipeline) |
| Repo / PR metadata strings | `ai_code_reviewer.models.metadata.build_repository_metadata`, `build_pull_request_metadata` (via the pipeline) |
| Prompt assembly | `ai_code_reviewer.models.pipeline.ReviewPipeline.run` |
| Local Qwen3 inference | `ai_code_reviewer.models.inference.ReviewModel` |
| JSON output parsing | `ai_code_reviewer.models.inference.ReviewModel._parse_response` (via `predict`) |

The only logic added by the demo is:

* a small synchronous `urllib`-based GitHub client (the project's main
  fetcher is async/batch-oriented and would not fit a Streamlit callback);
* shape conversion (`ReviewPrediction` -> UI-friendly dicts);
* render helpers in `demo/view.py`.

`outgoing_dependencies`, `incoming_dependencies`, and `metadata_files` are
left empty for the demo (the prompt then shows them as `"None"`). Hooking
the dataset's full repo-snapshot enrichment into a live request would require
downloading the snapshot zipball; that path is intentionally left to the
batch pipeline.

## Files

```
demo/
  __init__.py        package marker
  app.py             Streamlit entrypoint (eager model load + URL input)
  adapters.py        live GitHub fetch + reuse of ReviewPipeline/ReviewModel
  view.py            CSS + render helpers (cards, sidebar, diff, comments)
  demo_data.py       legacy mock fixture (no longer wired into the UI)
  README.md          this file
```

A compatibility shim is kept at the repo root: `demo_app.py` simply imports
and runs `demo.app.main`, so the original `streamlit run demo_app.py`
command keeps working.

## Example

```bash
GITHUB_TOKEN=ghp_xxx streamlit run demo/app.py
# In the UI, paste e.g.
#   https://github.com/psf/requests/pull/6789
# and click "Analyze PR".
```

## Troubleshooting

* **App opens, then shows `Local model error: Local inference requires the
  finetune extras`** — install with `pip install -e ".[finetune,demo]"`,
  or switch to `APR_BACKEND=ollama` (see "Quick start with Ollama" above).
* **`Ollama server not reachable at http://localhost:11434`** — start it
  with `ollama serve` (or `brew services start ollama`).
* **`Ollama model X is not pulled`** — run `ollama pull <model>` first.
* **`GitHub API HTTP 403 ... rate limit exceeded`** — set `GITHUB_TOKEN`.
* **`GitHub API HTTP 404`** — the PR is private or the URL is wrong; ensure
  your token has access.
* **Inference is very slow on Apple Silicon with the default backend** —
  switch to `APR_BACKEND=ollama` (typically 5-10x faster than transformers
  + MPS on M-series chips).
* **Model loads but no issues are ever found** — for `transformers` try
  lowering `APR_MAX_INPUT_TOKENS` (e.g. `8192`) or set `APR_LOAD_IN_4BIT=1`
  if you hit OOM. For `ollama`, try a stronger code model
  (e.g. `qwen2.5-coder:7b`).

## What is **not** in the demo

* No baseline-vs-finetuned comparison (the demo presents a single reviewer
  model, as required).
* No posting of comments back to GitHub.
* No fine-tuning UI; that pipeline lives in
  `src/ai_code_reviewer/finetuning/`.
