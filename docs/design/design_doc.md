# Automated Pull Request Reviewer: ML System Design

This document outlines the machine learning system design for Automated Pull Request Reviewer. It serves as our Contract of Work and ensures alignment across DS, ML, and Business roles.

## 1. Problem Definition

### 1.1 Context

Senior developers spend up to 20-30% of their time on code reviews. A significant portion of this effort is wasted on identifying repetitive, low-level issues that static analysis tools miss, but which are blocking for production (e.g., specific logic errors, security vulnerabilities in context, bad code practices).

Existing AI solutions (Copilot/ChatGPT) require sending proprietary code to external APIs, which violates data privacy policies in many enterprise environments.

To address these challenges, we introduce Automated Pull Request Reviewer (APR), an automated code review agent designed to act as a privacy-first quality gate. By leveraging machine learning models, our agent identifies blocking issues locally within the CI/CD pipeline. This solution effectively eliminates data leakage risks while significantly reducing the cognitive load on human reviewers by filtering out critical errors before they reach the senior team.

### 1.2 Functional Requirements

- **We will focus exclusively on Python repositories.** 
The initial version of the agent will only analyze changes made to standard Python scripts (`.py` files), ignoring Jupyter Notebooks and other languages.
- **We will prioritize semantic issues over style.** 
The model will assume that standard linters (like `black` or `flake8`) run in previous CI steps, so we will not flag formatting/style errors.
- **The model will solve a binary classification task.** 
For each code diff hunk, the system will output a probability of whether it contains a blocking issue or not.
- **The agent will generate a human-readable comment.** 
If a code issue is detected, the model will provide a descriptive message explaining the potential problem (e.g., explaining why a default mutable argument is dangerous).
- **The output will include precise line localization.** 
The API response will pinpoint the exact lines within the diff context where the error occurs, allowing inline comments in GitHub.

### 1.3 Key Stakeholders & Business Impact

Our solution delivers value to different roles within the organization:

**Junior/Middle Developers**

- **Accelerated Time-to-Merge**: Receive immediate feedback on blocking issues without waiting hours for a human review;
- **Learning & Compliance**: Quickly learn team standards and best practices through consistent, automated explanations, reducing the fear of submitting "bad code" to seniors.

**Senior Developers / Tech Leads**

- **Reduced Cognitive Load**: Automate the mundane task of catching repetitive, low-level issues (e.g., style violations, obvious bugs);
- **High-Value Allocation**: Shift focus from "nitpicking" to architectural design, complex logic, and mentoring, increasing overall team velocity.

**Security / Compliance Officers**

- **Zero Data Leakage**: Run ML inference locally (On-Premise), ensuring proprietary code never leaves the secure perimeter (unlike cloud-based LLMs);
- **Consistent Scanning**: Enforce critical security checks on every single Pull Request, eliminating human factors.

## 2. Metrics

### 2.1 Online Metrics

We aim to improve the efficiency and stability of the development process. The key metrics we track are:

- **Time-to-Merge (Cycle Time):** 
The total duration from creating a Pull Request to merging it into `main`. 
  - *Goal:* Decrease by 20% by reducing feedback loops on trivial issues.

- **First-Time Approval Rate:** 
The percentage of PRs that are approved by a human reviewer without requiring any changes.
  - *Goal:* Increase by 5%, as the agent catches blocking issues *before* human review starts.

- **Change Failure Rate (CFR):** 
The percentage of deployments causing a failure in production (e.g., hotfixes/reverts).
  - *Goal:* Decrease by 5% by catching `CHANGES_REQUESTED` level bugs (e.g., Security/Logic) early in CI.

- **Comments per Pull Request:** 
The average number of review comments per PR.
  - *Goal:* Decrease by 7% by catching issues automatically before human review.

### 2.2 Business Value

Senior developers spend 8-12 hours per week reviewing code ([Source](https://dev.to/sociilabs/the-40k-code-review-tax-why-manual-reviews-are-bleeding-your-engineering-budget-3485)). We expect our product to decrease the required time for code reviews by roughly **7%**, or by **35-50 minutes per person weekly**. With a salary of $200k/year, that translates to **$3k-4.2k/year per person in time savings alone**. On top of that, it will reduce back-and-forth between developers, allowing for significantly faster feature delivery.

- **Time-to-Merge (Cycle Time):** 
For an average team, time-to-merge is 73-141 hours. A decrease by 20% would save 14.6-28.2 hours of waiting per pull request. [Source](https://ru.scribd.com/document/983996371/LinearB-2026-Software-Engineering-Benchmarks-Report)

- **First-Time Approval Rate:**
For an average team, the first-time approval rate is 80%, or 1.2 review cycles per pull request. By increasing first-time approval rate by 5%, multiple hours if not days of waiting can be saved for every 20th pull request. [Source](https://codeclimate.com/blog/pull-request-reviews-for-enterprise-engineering?trk=public_post_comment-text)

- **Change Failure Rate (CFR):** 
For an average team, the change failure rate is 5-17%. A relative 5% reduction in CFR (multiplying by 0.95, i.e., a 0.25–0.85 percentage-point drop from a 5–17% baseline) would avoid one production failure per roughly 115–400 pull requests: (1 / (0.05\*0.17) = 115) and (1 / (0.05\*0.05) = 400). [Source](https://ru.scribd.com/document/983996371/LinearB-2026-Software-Engineering-Benchmarks-Report)

- **Comments per Pull Request:** 
An average team at Google receives 2 comments per pull request or 5 comments per 100 lines of code. Decrease by **7%** would save 0.14 comments per pull request or 1 comment every 285 lines of code. [Source](https://research.google/pubs/modern-code-review-a-case-study-at-google/)


### 2.3 Offline Metrics

**1. Classification Quality:**
We prioritize **Precision** over Recall because false positives annoy developers and reduce trust in the tool.
*   **Precision (Class 1 - Blocking Issue):** Target **> 70%**. We want to be sure that flagged code is truly problematic;
*   **Recall:** Target **> 50%**. Catching half of all critical bugs is a significant improvement;
*   **Macro F1-Score:** For overall model comparison.

**2. Comment Quality:**
To evaluate the relevance of generated comments against human reviews:
*   **BERTScore:** Measures semantic similarity between the model's output and the ground truth human comment (robust to paraphrasing);
*   **LLM-as-a-Judge:** We will use Qwen3 to rate the helpfulness of generated comments on a 1-5 scale.


## 3. Dataset

### 3.1 Language Popularity
For the initial MVP we will focus entirely on Python scripts (`.py` files) for simplicity and focus on the core functionality.

According to [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/2025/technology#most-popular-technologies-language-language), Python is the second most popular language across professional developers, and first among people who are learning to code, which shows it's potential to become a dominant language in the future. Moreover, ratings like [TIOBE](https://www.tiobe.com/tiobe-index/) and [PYPL](https://pypl.github.io/PYPL.html) puts python on the top of the most popular languages list, gaining popularity year over year.

Another factor towards focusing on Python is the fact that Python is the default choice by majority of LLMs itself (largely due to the language's popularity and ease of use) according to [Twist et al. (2026)](https://arxiv.org/pdf/2503.17181v3). This could simplify and speed up the modeling process for the initial MVP, while doesn't strictly make the architecture incompatible with other languages (just a bit more nuanced tuning required).

Given the above, we will focus on Python scripts (`.py` files, but excluding Jupyter Notebooks due to parsing complexity) for the initial MVP.

### 3.2 GH Archive Data
As the main source of PR comments data we will use [GH Archive](https://www.gharchive.org/). This dataset records every public event on GitHub, including Pull Request review comments. 

From this source we will extract:
- 
- 

To the best of our knowledge, there is no rate limiting on this dataset, we we could efficiently parallelize the data collection process.

### 3.3 GitHub API Data
Though GH Archive provides information about PR review comments, it lacks context around the PR and changed files themselves. For this we will utilize GitHub API to fetch additional information. More specifically, we will utilize two types of API: (1) REST API to fetch PR and changed files content, (2) GraphQL API to fetch PR metadata.

From GitHub REST API we will fetch:
-
-

From GitHub GraphQL API we will fetch:
-
-
-


### 3.4 Dependencies Resolution

#### 3.4.1 Incoming Dependencies


#### 3.4.2 Outgoing Dependencies

### 3.5 Data Preparation


## 4. Solution

### 4.1 Baseline Solution

> **TODO:** https://github.com/lolyhop/ai-code-reviewer/issues/3


### 4.2 Advanced Solution

> **TODO:** https://github.com/lolyhop/ai-code-reviewer/issues/4

### 4.3 Measurement

**1. Offline Evaluation:**
We measure the core performance of the Classification Model before any deployment.
*   **Precision (High Priority):** > 70%. We must minimize False Positives to avoid developer fatigue.
*   **Recall:** > 50% for critical categories.
*   **Semantic Similarity (Generated Comments):**
    *   **BERTScore:** Calculate similarity between generated comment and human ground truth.
    *   **LLM-Eval (Optional):** Use Qwen API to rate the helpfulness (1-5) of a sample of N generated comments.
    *   *Reference:* `docs/reports/5_evaluation.md` will contain the final metrics table.

**2. Success Criteria (Business Acceptance):**
> **TODO:** Define the exact Acceptance Thresholds (e.g., "Precision must be > 80% on the Golden Set to launch"). This decision will be made by the Business Unit after initial model training.

**3. Online Experimentation:**
Although out of scope for the MVP implementation, we propose a robust **A/B Testing Strategy** for future rollout:
*   **Randomization Unit:** Pull Request ID (hash based). 50% Control / 50% Treatment.
*   **Control Group:** Standard human review process (No Bot).
*   **Treatment Group:** AI-Reviewer posts comments automatically.
*   **Metric Comparison:** Measure difference in *Time-to-Merge* and *Comments per PR* between the two groups over 2 or more weeks.

## 5. Integration

**5.1 Demo Format**
We will showcase the agent's capabilities via a custom Web UI (using Streamlit or Gradio).
*   **Input:** URL of any public GitHub PR (e.g., `https://github.com/pandas-dev/pandas/pull/123`).
*   **Process:**
    1.  Fetch PR diff via API.
    2.  Run model inference.
    3.  Display the Diff with highlighted **Blocking Issues**.
*   **Output:** A clean, side-by-side view showing the "before" and "after" (with auto-generated comments).

**5.2 Final Deliverables**
*   **GitHub Repository:** Full source code (`src/`), organized documentation (`docs/`), and Kanban board (`Projects`).
*   **Design Documents:** `docs/design/design_doc.md`.
*   **CRISP-DM Report:** Consolidated PDF report (`docs/reports/final_report.pdf`).
*   **Video Demo:** A 5-minute walkthrough of the tool analyzing real-world PRs on GitHub.
