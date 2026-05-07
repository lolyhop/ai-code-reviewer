# Demo (`demo/`)

Streamlit UI for [Automated Pull Request Reviewer](../README.md): paste a PR URL, fetch via GitHub API, build prompts with `ReviewPipeline`, run the configured inference backend, render output.

## Quick start

```bash
pip install -e ".[finetune,demo]"
export GITHUB_TOKEN=ghp_...   # optional but avoids rate limits
streamlit run demo/app.py
```

→ [http://localhost:8501](http://localhost:8501)

On startup the model loads once (`streamlit.cache_resource`). If loading fails, the main UI is not shown and an error + hints appear.

## Backends (`APR_BACKEND`)

| Value | Use case | Notes |
|---|---|---|
| `transformers` | Default; GPU / HF weights | `pip install -e ".[finetune,demo]"` |
| `ollama` | Often faster on Apple Silicon (Metal / llama.cpp) | `brew install ollama`, `ollama pull …` |
| `openai` | Remote Chat Completions-compatible HTTP API | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` |

Prompt building and JSON parsing are shared; only generation differs.

`OPENAI_BASE_URL` is the base that exposes `POST …/chat/completions` (e.g. `https://api.openai.com/v1`). Same schema as OpenAI required.

### Ollama example

```bash
ollama serve &
ollama pull qwen2.5-coder:1.5b

APR_BACKEND=ollama APR_OLLAMA_MODEL=qwen2.5-coder:1.5b streamlit run demo/app.py
```

Startup hits `/api/tags` and sends a tiny chat to reduce cold start.

### OpenAI-compatible example

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1
export APR_OPENAI_MODEL=gpt-4o-mini
export APR_BACKEND=openai
streamlit run demo/app.py
```

Traffic goes to `OPENAI_BASE_URL`; only use if policy allows.

## Environment variables

### Common

| Variable | Default | Notes |
|---|---|---|
| `GITHUB_TOKEN` | — | Private PRs; higher rate limits on public repos |
| `APR_BACKEND` | `transformers` | `transformers`, `ollama`, `openai`. `openrouter` → treated as `transformers` |
| `APR_MAX_NEW_TOKENS` | `512` | Max new tokens per file |

### `transformers`

| Variable | Default |
|---|---|
| `APR_MODEL_NAME` | `Qwen/Qwen3-1.7B` |
| `APR_DEVICE` | `auto` |
| `APR_TORCH_DTYPE` | `bfloat16` |
| `APR_MAX_INPUT_TOKENS` | `16384` |
| `APR_LOAD_IN_4BIT` | `false` |

### `ollama`

| Variable | Default |
|---|---|
| `APR_OLLAMA_MODEL` | `qwen2.5-coder:1.5b` |
| `APR_OLLAMA_URL` | `http://localhost:11434` |

### `openai`

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | required |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `APR_OPENAI_MODEL` | `gpt-4o-mini` |
| `APR_OPENAI_JSON_OBJECT` | `1` → set `0` if `response_format: json_object` unsupported |

## Pipeline reuse

| Step | Module |
|---|---|
| Patch → `patched_content` | `ai_code_reviewer.dataset.patches.compute_patched_content` |
| Rows | `ReviewSample` |
| Retrieval + prompts | `ReviewPipeline.run` |
| Inference | `ReviewModel` (transformers) or wrappers in `demo/adapters.py` |
| Parse output | same JSON shape as `ReviewModel` |

Extra demo-only code: sync GitHub client (`urllib`), `ReviewPrediction` → UI dicts, `view.py` markup.

Demo leaves `outgoing_dependencies`, `incoming_dependencies`, `metadata_files` empty (prompt shows `"None"`). Full zipball enrichment stays in the batch pipeline.

## Files

```
demo/
  app.py          entrypoint
  adapters.py     GitHub + pipeline + backends
  view.py         layout/CSS
  demo_data.py    unused mock fixture
  README.md
```

Repo root may ship `demo_app.py` → `demo.app.main`.

## Troubleshooting

- **`Local inference requires the finetune extras`** — `pip install -e ".[finetune,demo]"` or `APR_BACKEND=ollama`
- **Ollama unreachable** — `ollama serve`
- **Model not pulled** — `ollama pull <name>`
- **GitHub 403 rate limit** — set `GITHUB_TOKEN`
- **GitHub 404** — wrong URL or token scope
- **Slow on Mac + transformers** — try `APR_BACKEND=ollama`
- **OOM / empty predictions** — lower `APR_MAX_INPUT_TOKENS`, or `APR_LOAD_IN_4BIT=1`, or a larger Ollama tag

Not implemented: posting comments to GitHub, training UI (`src/ai_code_reviewer/finetuning/`).
