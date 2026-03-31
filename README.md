# Automated Pull Request Reviewer

Automated Pull Request Reviewer is a LLM-based system for reviewing Python pull requests. The project focuses on detecting merge-blocking semantic issues in changed code, localizing them to the relevant lines, and generating human-readable review comments. The system is designed for on-premise usage, so proprietary code does not need to be sent to external APIs.

Our solution processes pull requests file by file, augments each changed file with repository-specific context, and uses a language model to identify correctness, safety, and maintainability issues that may be missed by standard static analysis tools.


## 📂 Project Structure

```text
.
├── data/                  # Raw, intermediate, and processed datasets
├── docs/                  # Design documents, reports, and methodology
├── notebooks/             # Research notebooks, experiments, and EDA
├── src/                   # Source code for data collection, preprocessing, retrieval, and inference
│   ├── data/              # Dataset construction and labeling pipeline
│   ├── retrieval/         # Repository context extraction and candidate generation
│   ├── models/            # Model wrappers, prompting, and fine-tuning code
│   ├── evaluation/        # Offline evaluation and metrics
│   └── app/               # Demo or interface code
├── tests/                 # Unit and integration tests
├── README.md              # Project overview
└── requirements.txt       # Python dependencies
```

> The exact structure may evolve during development as we finalize the pipeline and experiments.

## 👥 Team
| Role   | Name                      | Email                                                                                 | Responsibilities |
| ------ | ------------------------- | ------------------------------------------------------------------------------------- | ---------------- |
| **PM** | **Egor Chernobrovkin**    | *[e.chernobrovkin@innopolis.university](mailto:e.chernobrovkin@innopolis.university)* | Project management, task coordination |
| **DS** | **Nikita Tiurkov**        | *[n.tiurkov@innopolis.university](mailto:n.tiurkov@innopolis.university)*             | Dataset research, exploratory data analysis |
| **BA** | **Nurmukhammet Adagamov** | *[n.adagamov@innopolis.university](mailto:n.adagamov@innopolis.university)*           | Business analysis, requirements gathering |
| **ML** | **Ivan Ershov**           | *[i.ershov@innopolis.university](mailto:i.ershov@innopolis.university)*               | Model development, prompting, fine-tuning |
| **BU** | **Ruslan Gatiatullin**    | *[r.gatiatullin@innopolis.university](mailto:r.gatiatullin@innopolis.university)*     | Business validation, success criteria definition |
