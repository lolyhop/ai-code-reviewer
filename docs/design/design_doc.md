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

### 3.2 Data Sources

#### 3.2.1 GH Archive Data
As the main source of PR review comments data we use [GH Archive](https://www.gharchive.org/), which records every public event on GitHub as hourly NDJSON (gzip-compressed) files. 

From `PullRequestReviewCommentEvent` entries, we extract:
- **Review comment body**: The text of the inline code review comment written by a human reviewer.
- **Diff hunk**: The code context (surrounding lines) where the comment was posted, essential for understanding what code triggered the feedback.
- **Comment metadata**: File path, line numbers, and commit IDs (the snapshot commit where the comment was posted).

We filter to Python files only (`.py` extension), exclude comments on the old code version (`LEFT` side), and skip non-English comments and replies. To the best of our knowledge, there is no rate limiting on this dataset, allowing efficient parallel data collection.

#### 3.2.2 GitHub API Data
While GH Archive provides review comments, it lacks critical context about PR diffs and repository structure. We use GitHub's REST and GraphQL APIs to enrich this data.

**From GitHub REST API:**
- **Commit compare endpoint**: Fetches the unified diff (patch) and file metadata for all changed files between base and head commits, including file status (added/modified/deleted) and rename information.
- **Zipball downloads**: Downloads a complete snapshot of the repository at a specific commit, enabling extraction of file content, directory structure, and dependency metadata for both the main changed file and dependency files.

**From GitHub GraphQL API:**
- **PR-level metadata**: Pull request title and description (body), which provide context for understanding the change's intent.
- **Repository metadata**: Star count, used as a proxy for project maturity and adoption.
- **Review thread resolution**: Determines whether inline comments were marked as resolved, useful for filtering or prioritizing unresolved issues.

### 3.3 Dependencies Resolution
To provide comprehensive context around code changes, we resolve both incoming and outgoing dependencies. A diff patch alone shows changes in isolation; dependencies reveal how those changes cascade through the codebase (what uses the modified code) and what infrastructure the change relies on (what the modified code uses).

#### 3.3.1 Incoming Dependencies

**Definition**: Files in the repository that import or reference identifiers (functions, classes, variables) defined in the changed file.

**Why we need them**: When a developer modifies a function signature or class definition, downstream files using that code may break. By including incoming dependencies, the model can better understand the impact of changes and flag breaking modifications.

**Resolution process**: 
1. Extract symbols (function/class names) whose definitions or implementations changed in the main file using AST analysis of the patch.
2. Scan all Python files in the repository for word-boundary occurrences of these symbols.
3. Return matched files as incoming dependencies.

#### 3.3.2 Outgoing Dependencies

**Definition**: Files that the changed file imports from, either as direct imports or through attribute chains.

**Why we need them**: The modified code may be using outdated, deprecated, or poorly-designed functionality from other modules. By including outgoing dependencies, the model sees what patterns the change relies on and can flag anti-patterns or problematic interactions.

**Resolution process**:
1. Parse import statements (both absolute and relative imports) from the changed file.
2. Resolve each import to candidate file paths in the repository, accounting for package structure and source roots (e.g., `src/` layouts).
3. For each imported name, expand candidates based on observed attribute access patterns in the code (e.g., `module.submodule.function`).
4. Return matched files as outgoing dependencies.

### 3.4 Data Preparation

#### 3.4.1 Quality Validation
Quality validation is applied at multiple stages to ensure reliable training data:

**GH Archive Ingestion Filters:**
- **Event type**: Only `PullRequestReviewCommentEvent` records are retained.
- **File type**: Python files (`.py` extension) only; other languages and Jupyter Notebooks excluded.
- **Comment side**: Exclude comments on the LEFT side of diffs (outdated code version); include RIGHT side only (new code).
- **Reply comments**: Exclude comments that are replies to other comments (`in_reply_to_id` must be null).
- **Field presence**: Comment body and diff_hunk must both be non-empty.
- **Language detection**: ASCII character ratio must be ≥ **70%** (`IS_LIKELY_ENGLISH_THRESHOLD = 0.7`) to filter non-English reviews.

**Size Constraints:**
- **Single file**: Maximum **2 MB** (`FILE_MAX_BYTES = 2,097,152`) to skip corrupted or binary files.
- **Repository zipball**: Maximum **50 MB** (`ZIPBALL_MAX_BYTES = 52,428,800`) to prevent unbounded downloads.

**Post-Processing Validation:**
- **Null/NA removal**: Parquet export removes rows with null values in critical columns.
- **Path normalization**: Normalize and validate all paths against path traversal attacks before processing.
- **Non-empty diff**: Exclude files with empty `patched_content` (no actual changes).

#### 3.4.2 Context Compression
After exploratory data analysis, we apply the following limitations and transformations to fit content into LLM context windows while preserving semantic information:

**PR Metadata** (`pr_title` + `pr_body`):
- Truncate to first **500 characters** (~125 tokens) to capture the intent while removing verbose instructions or repetitive checklists.
- Strip HTML tags to reduce verbosity while preserving text content.

**Repository Metadata**:
- **README.md**: First **400 characters** (~100 tokens), HTML tags removed. Sufficient to convey project purpose without implementation details.
- **Configuration files** (requirements.txt, setup.py, pyproject.toml, setup.cfg): Extract viable parts via regex (project name, description, dependencies) and truncate to **500 characters** (~125 tokens) per file.
- **Repo metadata cap**: Limit total repository metadata to **4000 characters** (~1000 tokens) across all metadata files to prevent extreme outliers while preserving high-level project context.

**Repository File Tree**:
- Truncate to **2 levels of depth** (root directory and immediate children) to provide structural context without overwhelming detail.
- Exclude dot-directories (`.git`, `.github`, `.vscode`, etc.) which contain metadata irrelevant to code review.

**Main Changed File**:
- Hide function/method bodies without `+` or `-` lines (unmodified code), replacing with `[function body is hidden]` placeholder while preserving the signature.
- Replace multi-line docstrings exceeding **3 lines** with `[docstring hidden]`.
- Collapse consecutive comment blocks exceeding **3 lines** to the first 3 lines plus a marker `# ... [N comment lines hidden]`.
- Reduce runs of more than **2** consecutive blank lines to 2.
- No hard character truncation (to preserve essential context).

**Incoming and Outgoing Dependencies**:
- Include top **5 most relevant files** from each dependency type (preserves >75th percentile of PRs without changing distribution).
- Exclude configuration files (`__init__.py`, `setup.py`, `conftest.py`, `__version__.py`) which lack code-related context.
- Apply the same compression techniques as the main file (hidden bodies, collapsed comments, etc.).

#### 3.4.3 LLM-based Comment Labelling

Human review comments are inherently noisy — reviewers also leave stylistic suggestions, questions, compliments, and nitpicks. Treating all commented files as positive examples of "buggy code" would severely degrade label quality. Similarly, files with no comments are not guaranteed to be clean: a reviewer may have missed an issue or the file may not have been carefully reviewed at all.

To address both problems, we employ an **LLM-as-a-judge pipeline** (Qwen3 235B via Yandex Cloud Foundation Models) that processes every sample and produces structured labels.

**Comment Classification Taxonomy:**

To define the taxonomy in a data-driven way, we first sampled ~500 human review comments and asked a DeepSeek-v3.2 model to cluster them into coherent groups. After several dialogue iterations, we converged on a nine-category taxonomy:

| Category | Description |
|---|---|
| `blocking_issue` | Correctness bugs, security vulnerabilities, data-loss risks, crashes, race conditions |
| `performance` | Efficiency concerns, unnecessary allocations, algorithmic complexity |
| `best_practice` | Design patterns, idiomatic code, architectural suggestions, error-handling improvements |
| `style` | Formatting, naming conventions, import ordering, whitespace |
| `documentation` | Missing or incorrect docstrings, comments, type hints |
| `question` | Reviewer asking for clarification or explanation |
| `nitpick` | Minor optional observations that do not affect correctness or style |
| `praise` | Positive feedback, approval, compliments |
| `other` | Does not fit any category above |

**Two Prompt Variants:**
- **Classification prompt** (files *with* comments): the judge receives full code context (PR metadata, repo metadata, changed file, dependencies, file tree) plus the raw review comments, and outputs one category per comment.
- **Validation prompt** (files *without* comments): the judge receives the same code context and answers a single binary question — does this file contain a blocking issue? Only files validated as `false` become clean negative examples.

This two-prompt design ensures both positive and negative labels are actively validated rather than assumed.

#### 3.4.4 Data Sampling

The core challenge is severe class imbalance: blocking issues represent only **~11% of all comments**. Naive oversampling or loss reweighting would create artificial patterns. Instead, we apply **selective positive class inclusion** and **high-quality negative augmentation**:

**Positive Class Strategy:**
- **Blocking issues** (`blocking_issue`): **100% inclusion** — these are our target class; every positive example is valuable.
- **Performance** (`performance`): **100% inclusion** — rare but highly actionable; all instances retained.
- **Best practice** (`best_practice`): **50% random sampling** — largest category; prevents overwhelming negatives with procedural feedback.
- Rationale: Sampling maintains signal while controlling dataset size and preventing memorization of common patterns.

**Negative Class Definition:**
- Files with **zero human comments** AND **LLM-validated** to contain no blocking issues (high-confidence negatives).
- This avoids the common pitfall of training on unlabeled data that may actually contain bugs.

**No-Comment File Augmentation:**
- When enabled (`INCLUDE_NO_COMMENT_FILES = True`), for each snapshot commit, randomly add uncommented Python files from the same commit.
- Cap: min(count of commented files, available uncommented candidates) — prevents imbalance within snapshots.

**Snapshot Commit Filtering:**
- Retain top **2,000 snapshot commits** globally (`SNAPSHOT_COMMITS_TO_KEEP = 2000`), ranked by comment count (descending).
- Break ties deterministically by `(repo_name, pr_number, commit)` lexicographic order to ensure reproducibility.
- Rationale: Focuses on high-activity commits while maintaining a diverse, manageable dataset.

#### 3.4.5 Train/Validation/Test Split

Preventing data leakage is critical; we use **multi-level isolation** to ensure clean evaluation:

**No-Leakage Guarantees:**
1. **PR-level isolation**: All files from the same Pull Request (including multiple snapshot commits) go to the same split (train or val or test). Prevents the model from memorizing PR-specific patterns.
2. **Repository-level holdout**: Every repository that appears in the test set is **completely removed** from the training and validation pools. Prevents the model from learning repository-specific coding styles or idioms.

**Test Set Construction:**
- **Size**: 100 pull requests with at least one file containing ≥1 `blocking_issue` comment.
- **Positives**: Files with ≥1 `blocking_issue` comment (128 files).
- **Negatives**: Files from the same PRs with zero comments and LLM-verified to contain no blocking issues (164 files).
- **Ratio**: 43.8% positive, 56.2% negative — realistic and challenging.

**Train/Validation Split:**
- **Ratio**: 80% training, 20% validation.
- **Stratification**: Stratified by label (blocking issue present or absent) to maintain class distribution across splits.
- **Total train samples**: 1,913 (687 positive, 1,226 negative, 35.9% positive)
- **Total val samples**: 536 (162 positive, 374 negative, 30.2% positive)


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
