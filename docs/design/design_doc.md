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
  - *Goal:* Increase by 25%, as the agent catches blocking issues *before* human review starts.

- **Change Failure Rate (CFR):** 
The percentage of deployments causing a failure in production (e.g., hotfixes/reverts).
  - *Goal:* Decrease by 5% by catching `CHANGES_REQUESTED` level bugs (e.g., Security/Logic) early in CI.

- **Comments per Pull Request:** 
The average number of review comments per PR.
  - *Goal:* Decrease by 7% by catching issues automatically before human review.

### 2.2 Business Value

Business value of engineering metrics

[source 1](https://research.google/pubs/modern-code-review-a-case-study-at-google/)

[source 2](https://research.google/pubs/modern-code-review-a-case-study-at-google/)


- **Time-to-Merge (Cycle Time):** 
For a fair team takes 73-141 hours on average. A decrease by 20% would save 14.6-28.2 hours per pull request for a team.

- **Change Failure Rate (CFR):** 
For a fair team is 5-17% on average. A decrease by 5% would avoid one production failure per 115-400 pull requests (which is easily achiavable in a few months).

- **Comments per Pull Request:** 
An average team at Google has 2 comments per pull request. Decrease by 7% would save 0.14 comments per pull request or 1 comment every 350 lines of code.


### 2.2 Offline Metrics

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

> **TODO:** https://github.com/lolyhop/ai-code-reviewer/issues/2

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
