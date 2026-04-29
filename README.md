# Automated Pull Request Reviewer

Automated Pull Request Reviewer is a LLM-based system for reviewing Python pull requests. The project focuses on detecting merge-blocking semantic issues in changed code, localizing them to the relevant lines, and generating human-readable review comments. The system is designed for on-premise usage, so proprietary code does not need to be sent to external APIs.

Our solution processes pull requests file by file, augments each changed file with repository-specific context, and uses a language model to identify correctness, safety, and maintainability issues that may be missed by standard static analysis tools.


## Key Documents

| Document | Description |
|---|---|
| [`docs/design/design_doc.md`](docs/design/design_doc.md) | ML System Design: end-to-end pipeline, context construction, data labeling methodology, and evaluation strategy |
| [`docs/design/architecture.md`](docs/design/architecture.md) | Solution architecture: system overview, methodology, data pipeline, and limitations |
| [`docs/design/baseline_model.md`](docs/design/baseline_model.md) | Baseline model selection: hardware constraints, candidate comparison (LiveCodeBench), and rationale for choosing Qwen3-1.7B |
| [`docs/reports/comment_quality_evaluation.md`](docs/reports/comment_quality_evaluation.md) | Evaluation methodology: BERTScore vs ROUGE-L vs LLM-as-Judge, CodeBERT backbone choice, and final LLM-Judge framework with bias mitigation |

## 📂 Project Structure

```text
.
├── data/                          # Raw, intermediate, and processed datasets
├── docs/                          # Design documents, reports, and methodology
│   ├── design/                    # Architecture, baseline model, design doc
│   └── reports/                   # Evaluation methodology reports
├── notebooks/                     # Research notebooks, experiments, and EDA
├── src/
│   ├── ai_code_reviewer/          # Main package
│   │   ├── dataset/               # Dataset construction from GH Archive & GitHub API
│   │   ├── data_processing/       # Data cleaning & LLM-based labeling pipeline
│   │   ├── models/                # Inference, prompting, retrieval, and pipeline
│   │   └── finetuning/            # Supervised fine-tuning (qwen3-specific)
│   └── evaluation/                # Offline evaluation metrics (BERTScore, IoU, F1)
├── pyproject.toml                 # Build config and dependencies
├── README.md                      # Project overview
└── requirements.txt               # Evaluation dependencies
```

## 👥 Team
| Role   | Name                      | Email                                                                                 | Responsibilities |
| ------ | ------------------------- | ------------------------------------------------------------------------------------- | ---------------- |
| **PM** | **Egor Chernobrovkin**    | *[e.chernobrovkin@innopolis.university](mailto:e.chernobrovkin@innopolis.university)* | Project management, task coordination |
| **DS** | **Nikita Tiurkov**        | *[n.tiurkov@innopolis.university](mailto:n.tiurkov@innopolis.university)*             | Dataset research, exploratory data analysis |
| **BA** | **Nurmukhammet Adagamov** | *[n.adagamov@innopolis.university](mailto:n.adagamov@innopolis.university)*           | Business analysis, requirements gathering |
| **ML** | **Ivan Ershov**           | *[i.ershov@innopolis.university](mailto:i.ershov@innopolis.university)*               | Model development, prompting, fine-tuning |
| **BU** | **Ruslan Gatiatullin**    | *[r.gatiatullin@innopolis.university](mailto:r.gatiatullin@innopolis.university)*     | Business validation, success criteria definition |
