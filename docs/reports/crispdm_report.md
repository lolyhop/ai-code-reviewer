# CRISP-DM Report: Automated Pull Request Reviewer

## 1. Business Understanding

### 1.1 Business Objectives

Modern software teams spend a substantial amount of engineering time on pull request reviews. Senior developers and tech leads are expected to detect correctness issues, security risks, maintainability problems, and violations of internal engineering practices before code reaches production. However, a significant share of this effort is spent on repetitive issues that could be detected before human review begins.

The business objective of this project is to design an automated pull request review system that reduces the manual burden on senior reviewers while preserving code quality and data privacy. The proposed system, **Automated Pull Request Reviewer (APR)**, is intended to act as a privacy-first quality gate inside the CI/CD pipeline. It analyzes Python pull requests locally and flags potentially blocking issues before the pull request reaches a human reviewer.

For the initial MVP, the scope is intentionally limited to **Python repositories** and standard `.py` files. Jupyter Notebooks, non-Python files, formatting issues, and purely stylistic comments are considered out of scope. This restriction keeps the project focused on semantic review problems that are more valuable from a business perspective and less likely to overlap with existing linters such as `black`, `flake8`, or similar static analysis tools.

The main business goals and their measurable success indicators are summarized below.

| Business Goal | Metric / Indicator | Definition | Target |
|---|---|---|---:|
| Decrease pull request cycle time | **Time-to-Merge / Cycle Time** | Time between pull request creation and merge into the main branch | Reduce by ~20% |
| Improve review readiness | **First-Time Approval Rate** | Share of pull requests approved by a human reviewer without requiring changes after the first review round | Increase by ~5% |
| Reduce production risk | **Change Failure Rate** | Share of deployments that cause production incidents, rollbacks, hotfixes, or other immediate remediation | Reduce by ~5% relative |
| Reduce review noise | **Comments per Pull Request** | Average number of human review comments left on a pull request | Reduce by ~7% |
| Preserve code privacy | **Local / On-Premise Inference** | Whether proprietary code leaves the organization’s secure infrastructure during model inference | Must remain local/on-premise |

These targets reflect the intended business role of APR: the system is not expected to replace human reviewers, but to act as a pre-review quality gate that catches a useful subset of blocking issues with high precision and minimal developer disruption.

#### Expected Business Value

The expected value of APR comes primarily from reducing repetitive review effort and shortening feedback loops in the pull request process. One industry estimate suggests that senior developers may spend **8–12 hours per week** on code review work ([source](https://dev.to/sociilabs/the-40k-code-review-tax-why-manual-reviews-are-bleeding-your-engineering-budget-3485)). If APR reduces review noise and repeated review work by approximately **7%**, proxied by comments per pull request and review-cycle count, this corresponds to roughly **35–50 minutes saved per senior developer per week**. Assuming a senior developer salary of **$200k/year**, this translates to approximately **$3k–4.2k/year per person** in time savings alone.

The second source of value is faster delivery. Engineering benchmark data reports typical time-to-merge values in the range of **73–141 hours** for average teams ([LinearB benchmark report](https://ru.scribd.com/document/983996371/LinearB-2026-Software-Engineering-Benchmarks-Report)). A **20% reduction** in time-to-merge would therefore save approximately **14.6–28.2 hours of waiting time per pull request**, improving engineering throughput and reducing developer idle time.

APR may also improve review readiness by increasing the share of pull requests that pass human review without additional change cycles. Every avoided review cycle reduces waiting time, context switching, and coordination overhead. Industry discussion around enterprise pull request reviews commonly treats first-time approval and review-cycle count as important indicators of review efficiency ([Code Climate](https://codeclimate.com/blog/pull-request-reviews-for-enterprise-engineering?trk=public_post_comment-text)).

Another expected benefit is improved production stability. If APR helps reduce Change Failure Rate by **5% relative**, and the baseline CFR is between **5–17%**, this corresponds to a decrease of approximately **0.25–0.85 percentage points**. In practical terms, this means avoiding roughly one production failure per **115–400 pull requests**, depending on the initial failure rate ([LinearB benchmark report](https://ru.scribd.com/document/983996371/LinearB-2026-Software-Engineering-Benchmarks-Report)).

Finally, APR can reduce review noise. Google’s study of modern code review reports that an average change receives around **2 comments per review** and about **5 comments per 100 lines of code** ([Google code review case study](https://research.google/pubs/modern-code-review-a-case-study-at-google/)). A **7% reduction** in comments per pull request would save approximately **0.14 comments per PR**, or about one avoided comment per **285 lines of reviewed code**. While this effect is smaller than time-to-merge reduction, it supports the broader goal of reducing repetitive review friction.

---

### 1.2 Situation Assessment

The current pull request review process relies heavily on human reviewers, especially senior developers and tech leads. While human review remains essential for architectural decisions, complex reasoning, and knowledge sharing, it is inefficient when reviewers repeatedly spend time identifying similar low-level or routine issues. Modern code review is a standard practice in both open-source and industrial software development, and large-scale studies such as [Google’s modern code review case study](https://research.google/pubs/modern-code-review-a-case-study-at-google/) show that review systems operate at very high volume and are deeply integrated into engineering workflows.

First, manual review creates **high opportunity cost**. Senior developers are among the most expensive engineering resources, and their time is often better spent on design decisions, mentoring, and solving complex technical problems. When they spend a large share of review time on repetitive issues, the organization loses engineering capacity. This is especially important because code review is not a rare event: in mature engineering organizations it is a continuous process attached to almost every code change. For example, [Google’s review workflow analysis](https://research.google/pubs/modern-code-review-a-case-study-at-google/) examines review practices across millions of reviewed changes, illustrating the operational scale of review work in large software organizations.

Second, the review process introduces **feedback latency**. Developers may wait hours or days before receiving comments on problems that could have been detected automatically. This increases pull request cycle time and slows feature delivery. Engineering benchmark reports commonly track cycle time and pull request metrics as core delivery indicators, and [LinearB’s software engineering benchmarks](https://linearb.io/resources/software-engineering-benchmarks-report) treat metrics such as Cycle Time, PR Size, and Change Failure Rate as part of delivery performance measurement.

Third, review quality can be **inconsistent**. Human reviewers may miss issues due to fatigue, time pressure, context switching, or differences in personal review style. As a result, some blocking issues may only be discovered later in QA, staging, or production. This directly connects the project to reliability metrics such as **Change Failure Rate**, which [DORA defines as the share of deployments that require remediation after release](https://dora.dev/guides/dora-metrics/), for example through rollback, hotfix, or other immediate intervention.

Fourth, many existing AI-based code review or coding assistant solutions are difficult to adopt in enterprise environments because they require sending proprietary source code to external APIs. This creates a serious **data privacy and compliance risk**, especially for companies working with sensitive business logic, internal infrastructure, or customer data. Developer trust is also a practical concern: the [2025 Stack Overflow Developer Survey](https://survey.stackoverflow.co/2025/) shows broad AI adoption among developers, but also highlights that trust in AI-generated output remains limited. This supports the need for a conservative, precision-oriented system with human oversight rather than a fully autonomous reviewer.

The proposed APR system addresses these constraints by running inference locally within the organization’s infrastructure. The system is designed to integrate into the CI/CD process and review pull request changes before human reviewers begin their work. This makes the system a **pre-review quality gate** rather than a replacement for human reviewers: its goal is to catch a useful subset of blocking issues early, reduce repetitive review effort, and preserve privacy by ensuring that proprietary code does not leave the secure environment.

#### Available Resources

The project assumes access to the following resources:

- a dataset constructed from public GitHub pull request review activity;
- GH Archive data containing pull request review comment events;
- GitHub REST and GraphQL APIs for enriching review samples with pull request metadata, repository metadata, diffs, and repository snapshots;
- an on-premise or local GPU environment for model inference and fine-tuning;
- a single **NVIDIA A100 80 GB** GPU for experimentation with compact code-oriented language models;
- LLM-based labeling support for cleaning noisy human review data;
- engineering knowledge about Python code review practices and blocking issue categories.

#### Constraints

The project also has several important constraints:

- The MVP only supports **Python `.py` files**;
- The system does not review notebooks, generated files, vendored code, or non-Python changes;
- Formatting and style issues are out of scope because they are expected to be handled by linters;
- The model focuses on **blocking semantic issues**, such as correctness bugs, security vulnerabilities, data-loss risks, crashes, race conditions, and serious maintainability problems;
- The system must operate under a limited context window, so repository context must be compressed and selected carefully;
- Training labels are noisy because human review comments include questions, suggestions, style comments, praise, and non-blocking feedback;
- False positives are particularly costly because unnecessary bot comments can reduce developer trust.

---

### 1.3 Data Mining Goals

The business objective is translated into a supervised machine learning task. The core data mining goal is to build a model that can analyze a Python pull request change and decide whether it contains a blocking issue that should be raised before human review.

For the MVP, the primary task is formulated as a **binary classification problem** at the changed-file or diff-hunk level:

> Given a Python code change and its surrounding context, predict whether the change contains a blocking issue.

The positive class represents code changes that contain a blocking issue, such as:

- correctness bugs;
- security vulnerabilities;
- data-loss risks;
- crashes;
- race conditions;
- serious logic errors;
- serious maintainability problems with merge-blocking impact.

The negative class represents code changes that do not contain a blocking issue and should not trigger an automated review comment.

In addition to binary classification, the system should also support two secondary outputs:

1. **Line localization**  
   If an issue is detected, the model should identify the affected line or line range inside the diff.

2. **Human-readable review comment**  
   If an issue is detected, the model should generate a concise explanation that can be shown to the developer as an inline review comment.

Therefore, the full modeling objective contains three related tasks:

| Task | Output | Purpose |
|---|---|---|
| Blocking issue detection | Probability / binary label | Decide whether the file or diff hunk should be flagged |
| Line localization | Affected line range | Enable inline pull request comments |
| Comment generation | Short review message | Explain the issue to the developer |

For the initial classification objective, the most important offline metric is **precision for the positive class**. This reflects the business requirement that false positive comments are costly: if the tool posts many incorrect warnings, developers will stop trusting it.

The target offline metrics are:

| Metric | Target | Rationale |
|---|---:|---|
| Precision for blocking issues | > 70% | Minimize false positives and developer fatigue |
| Recall for blocking issues | > 40% | Catch a meaningful share of serious issues |
| Macro F1-score | Used for comparison | Balance performance across both classes |
| Localization quality | IoU-based evaluation | Check whether predicted line spans match actual issue locations |
| Comment quality | BERTScore / LLM-as-a-judge | Estimate usefulness of generated review comments |

The main data mining success criterion is not perfect bug detection. Instead, the goal is to produce a high-precision assistant that can reliably catch a useful subset of blocking issues before human review. In the business context, even partial automation is valuable if it reduces repetitive review work without creating excessive noise.

---

### 1.4 Project Plan

The project follows the CRISP-DM methodology and is organized into several stages. The initial goal is to build and evaluate an MVP version of the Automated Pull Request Reviewer for Python repositories.

#### Phase 1: Business Understanding

The first phase defines the business problem, project scope, stakeholders, and success criteria. The main decision in this phase is to focus on a privacy-preserving pull request review assistant for Python code, with emphasis on blocking semantic issues rather than formatting or style comments.

Key outputs:

- business objectives;
- stakeholder analysis;
- project scope;
- target business and offline metrics;
- initial risk assessment.

#### Phase 2: Data Understanding

The second phase focuses on understanding the available pull request review data. The project uses GH Archive as the main source of review comment events and GitHub APIs to enrich the data with pull request metadata, repository structure, diffs, and repository snapshots.

Key analysis questions include:

- how many usable Python pull request review comments can be collected;
- what share of comments correspond to blocking issues;
- how noisy human review comments are;
- how long review contexts are after compression;
- how often dependencies and usage-site context are available;
- whether the planned model context window is sufficient.

Key outputs:

- exploratory data analysis;
- comment taxonomy;
- analysis of class imbalance;
- review-context length statistics;
- feasibility check for model context length and hardware constraints.

#### Phase 3: Data Preparation

The third phase converts raw GitHub data into a model-ready dataset. Since raw human review comments are noisy, the project applies an LLM-as-a-judge pipeline to classify review comments and validate clean negative examples.

Main preparation steps:

- filter GH Archive events to Python pull request review comments;
- enrich samples through GitHub REST and GraphQL APIs;
- reconstruct changed file context and repository context;
- resolve incoming and outgoing dependencies;
- compress long context to fit model limits;
- classify human comments into a structured taxonomy;
- construct positive and negative examples;
- apply PR-level and repository-level split isolation to prevent leakage.

Key outputs:

- cleaned training dataset;
- validation set;
- held-out test set;
- blocking issue labels;
- line-level annotations;
- model-ready prompts.

#### Phase 4: Modeling

The modeling phase builds the first versions of the review model. The baseline model is a compact code-oriented LLM used in instruction-following mode. The advanced solution fine-tunes the selected model family on the prepared dataset.

Planned modeling steps:

- select a compact code-capable baseline model;
- run zero-shot inference on the validation set;
- parse model outputs into structured predictions;
- fine-tune the selected model on the training set;
- compare the baseline and fine-tuned versions using the same evaluation protocol;
- perform error analysis on false positives, missed blocking issues, localization failures, and low-quality comments.

The current baseline candidate is **Qwen3-1.7B**, selected because it offers a strong balance of model size, code reasoning capability, long-context support, and feasibility on the available A100 80 GB GPU. The detailed comparison with alternative candidates such as OpenCoder-1.5B-Instruct, DeepSeek-Coder-1.3B-Instruct, and Yi-Coder-1.5B-Chat is documented in the [baseline model selection document](https://github.com/lolyhop/ai-code-reviewer/blob/main/docs/design/baseline_model.md).

Key outputs:

- baseline inference pipeline;
- fine-tuned model candidate;
- validation metrics;
- error analysis;
- selected model for MVP integration.

#### Phase 5: Evaluation

The evaluation phase determines whether the model satisfies the project’s offline and business-oriented success criteria.

Evaluation includes:

- file-level blocking issue detection quality;
- precision, recall, and F1-score;
- line localization quality using IoU-based metrics;
- generated comment quality using BERTScore and optional LLM-as-a-judge;
- qualitative review of representative errors;
- assessment of whether the model is suitable for MVP deployment.

The main launch condition is high enough precision to avoid damaging developer trust. Recall is important, but the system is intended to assist reviewers rather than replace them, so it is acceptable for the MVP to catch only a subset of blocking issues as long as the flagged comments are useful and accurate.

Key outputs:

- final offline evaluation table;
- qualitative error analysis;
- decision on MVP readiness;
- recommendations for next iteration.

#### Phase 6: Deployment / Integration Plan

The MVP is planned as a local or on-premise pull request review service. For demonstration purposes, it will be integrated into a simple web interface that accepts a public GitHub pull request URL, fetches the diff, runs model inference, and displays detected blocking issues with inline comments.

The implementation and integration plan follows the system pipeline described in the [solution architecture document](https://github.com/lolyhop/ai-code-reviewer/blob/main/docs/design/architecture.md), where the pull request is processed at the file level, enriched with repository context, passed to the review model, and converted into structured review outputs.

Future production integration would place the system inside the CI/CD pipeline or GitHub review workflow.

Key outputs:

- demo web UI;
- inference pipeline;
- structured API response format;
- documentation;
- final CRISP-DM report;
- video demo.

#### Timeline

A realistic MVP timeline is approximately **8–12 weeks**:

| Phase | Duration | Main Output |
|---|---:|---|
| Business Understanding | 1 week | Problem definition and success criteria |
| Data Understanding | 1–2 weeks | EDA and feasibility analysis |
| Data Preparation | 2–3 weeks | Clean model-ready dataset |
| Modeling | 2 weeks | Baseline and fine-tuned model experiments |
| Evaluation | 2 weeks | Final metrics and error analysis |
| Integration / Demo | 2 weeks | Web demo and final report |

This plan is designed to produce a working MVP rather than a fully autonomous production reviewer. The system is expected to assist human reviewers by catching a useful subset of blocking issues while maintaining high precision and preserving code privacy.

---

## 2. Data Understanding

### 2.1 Data Collection

TODO

### 2.2 Data Description

TODO

### 2.3 Data Exploration

TODO

### 2.4 Data Quality

TODO

---

## 3. Data Preparation

### 3.1 Data Cleaning

TODO

### 3.2 Context Compression

TODO

### 3.3 LLM-based Labeling

TODO

### 3.4 Feature Construction

TODO

### 3.5 Train/Validation/Test Split

TODO

---

## 4. Modeling

### 4.1 Modeling Technique Selection

TODO

### 4.2 Baseline Model

TODO

### 4.3 Fine-tuned Model

TODO

### 4.4 Baseline Results

TODO

---

## 5. Evaluation

### 5.1 Evaluation Methodology

TODO

### 5.2 Offline Evaluation Results

TODO

### 5.3 LLM-as-Judge Evaluation

TODO

### 5.4 Review Process

TODO

---

## 6. Deployment

### 6.1 Deployment Plan

TODO

### 6.2 Monitoring and Maintenance

TODO

### 6.3 Final Report

TODO

---

## References

1. SociiLabs. *The $40K Code Review Tax: Why Manual Reviews Are Bleeding Your Engineering Budget*.  
   https://dev.to/sociilabs/the-40k-code-review-tax-why-manual-reviews-are-bleeding-your-engineering-budget-3485

2. LinearB. *Software Engineering Benchmarks Report*.  
   https://ru.scribd.com/document/983996371/LinearB-2026-Software-Engineering-Benchmarks-Report

3. Code Climate. *Pull Request Reviews for Enterprise Engineering*.  
   https://codeclimate.com/blog/pull-request-reviews-for-enterprise-engineering

4. Google Research. *Modern Code Review: A Case Study at Google*.  
   https://research.google/pubs/modern-code-review-a-case-study-at-google/

5. DORA. *DORA Metrics*.  
   https://dora.dev/guides/dora-metrics/

6. Stack Overflow. *Stack Overflow Developer Survey 2025*.  
   https://survey.stackoverflow.co/2025/

7. Project Documentation. *Baseline Model Selection for Automated Pull Request Reviewer*.  
   https://github.com/lolyhop/ai-code-reviewer/blob/main/docs/design/baseline_model.md

8. Project Documentation. *Automated Pull Request Reviewer Solution Architecture*.  
   https://github.com/lolyhop/ai-code-reviewer/blob/main/docs/design/architecture.md
