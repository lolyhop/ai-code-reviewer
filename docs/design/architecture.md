# Solution Design

In this document, we outline how our Automated Pull Request Reviewer works. We describe what the system takes as input, how it processes pull requests, how it builds context for each changed file, what data is needed for the pipeline, and how we evaluate the quality of the solution. We also detail how the baseline and advanced versions of the system are organized and what limitations this approach has.

## 1. System Overview

Our Automated Pull Request Reviewer is a retrieval-augmented LLM system for reviewing Python pull requests. The system takes a pull request as input, extracts its metadata and changed Python files, and processes each changed file independently. For every file, it builds a structured review context that combines local code changes with additional repository context, then asks a language model to identify blocking issues in the changed lines.

The system outputs a structured review result for each analyzed file. This result includes whether a blocking issue was detected, the affected line range, and a short human-readable comment describing the problem. File-level predictions are then aggregated into a final pull request review output.

The system is designed for Python repositories only and focuses on semantic issues, such as correctness, safety, and maintainability problems with merge-blocking impact. Style and formatting issues are considered out of scope, since they are expected to be handled by standard linters earlier in the CI pipeline.

To keep the design consistent, both the baseline and advanced versions of the solution follow the same high-level pipeline. They differ in the strength of the review model and in the quality of context selection, but not in the overall system structure.

## 2. Methodology

This section describes the end-to-end review pipeline used by our system. Since both the baseline and advanced solutions follow the same architecture, we define a single pipeline and later specify which components differ between the two setups.

### 2.1 Pull request processing unit

The system processes a pull request at the file level. Each changed Python file is reviewed independently, and the final pull request output is obtained by aggregating file-level predictions. We use file-level processing for two reasons. First, a full pull request may contain too many changed files to fit into the model context. Second, many review comments are attached to a specific file and a specific changed line range, so file-level processing is a natural unit for prediction and evaluation. Only standard Python source files (`.py`) are processed. Non-Python files, notebooks, generated files, and vendored code are ignored.

### 2.2 Input Collection

For each pull request, the system collects three groups of inputs:

1. **Pull request metadata**
2. **Changed file data**
3. **Repository-level context**

The exact contents of these inputs and the way they are transformed into model-ready context are described in Section 2.3.

### 2.3 Processing Unit Context Construction

For each changed Python file, the system builds a structured review context that combines local code changes with selected repository context. This context is then passed to the review model.

The local context is built from the changed file itself. It includes the file path and the patched file content, where additions and deletions are already represented as part of the diff. To reduce context size, the system truncates unchanged parts of the file. In particular, it removes the bodies of functions and methods that were not modified in the current patch. However, this truncation is applied only if these code blocks do not reference functions, methods, classes, or constants used in the changed code. This allows the system to save context space while preserving potentially relevant dependencies needed for review.

To enrich the local file context, the system also retrieves code from repository files imported by the changed file. First, it parses the imports in the changed file and keeps only those that resolve to Python modules inside the same repository. Then it checks which imported symbols are actually used in the modified code and retrieves their definitions, such as function, class, or constant definitions. To keep retrieval bounded, the system includes at most three such snippets per changed file.

The system also retrieves code snippets from repository files that use functions or classes modified in the changed file. To do this, it extracts changed top-level functions and classes and searches the repository for their usage sites. Explicit imports, qualified calls, and direct calls with matching imports are treated as evidence of usage. For each matched file, the system extracts a compact snippet, usually the function or method that contains the reference. At most three such snippets are added for each changed file.

In addition to retrieved code snippets, the system preprocesses a small set of repository-level files, including `README.md`, `CONTRIBUTING.md`, `requirements.txt`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, and others. In this preprocessing step, an LLM extracts repository-specific rules, conventions, and dependency information that may be useful for reviewing the pull request. These files are not passed to the review prompt in raw form. Instead, the extracted information is converted into a compact summary that can be included in the model context.

The final prompt for a single changed file contains pull request metadata, the repository summary, the local file context, code snippets from repository files imported by the changed file, code snippets from repository files that use functions or classes modified in the changed file, and the task instructions for the review model. If no additional repository code context is found, the system falls back to using only the pull request metadata, repository summary, and local file context.

### 2.4 Baseline and Advanced Setups

Both the baseline and advanced versions of the system use the same review pipeline described above. The difference between them lies in the review model itself.

The baseline setup uses the selected base model without additional domain-specific fine-tuning. It relies on the prompt structure, repository context construction, and retrieved code snippets to generate review predictions.

The advanced setup keeps the same pipeline, but additionally fine-tunes the review model on data collected from real pull request reviews. The goal of this stage is to adapt the model to the target task and improve its ability to detect blocking issues, localize them more precisely, and generate more useful review comments.

The training data for the advanced setup are constructed using the labeling procedure described in Section 3. In particular, we aim to build a cleaner supervision set from human review data by validating changed files and reducing label noise caused by both false positives and false negatives in raw human annotations.

## 3. Data

This section describes the data required to support the review pipeline.

### 3.1 Raw Data

For each pull request, the system requires pull request metadata, changed file data, human review annotations, and the repository snapshot at pull request time. Pull request metadata includes the title, description, and labels. Changed file data includes the repository name, pull request number, commit SHA, changed file paths, patched file content, and changed line ranges. Human review annotations include review comments together with their line annotations and, when available, their resolution status. The repository snapshot is needed to retrieve additional context, such as local imports, usage sites of modified functions or classes, and repository-level configuration files.

### 3.2 Evaluation Set Construction

Before training and evaluating the review model, we first need to construct a reliable evaluation set from the raw pull request review data. The same validation procedure can later be applied, at larger scale, to build a cleaner training set for advanced model fine-tuning.

The raw pull request review data cannot be used directly as a reliable evaluation set. Human review comments are a useful source of supervision, but they are noisy in two important ways. First, not every human comment corresponds to a merge-blocking issue: some comments are stylistic, organizational, or preference-based. Second, the absence of a human comment does not guarantee that the changed file contains no blocking issue, since reviewers may miss problems or choose not to comment on every affected file.

To address this, we construct a separate evaluation set from the raw dataset. We first randomly sample 100 pull requests from Python repositories in the initial collected corpus. These pull requests form the basis of the first evaluation set used to measure model quality.

For each sampled pull request, we examine every changed Python file, including both files with human comments and files without comments. We then use a stronger external LLM as a validation tool to review each changed file and determine whether it contains a blocking issue. This validation step is performed at the file level and is intended to reduce both types of label noise present in human review data:

- **False positives in human labels:** files with human comments that do not actually contain merge-blocking issues;
- **False negatives in human labels:** files without human comments that still contain merge-blocking issues.

As a result, the evaluation set is not defined purely by the presence or absence of human comments. Instead, each changed file in the sampled pull requests is validated for the presence of blocking issues. This allows us to keep only relevant blocking comments in positive examples and to verify negative examples with much higher confidence.

The final evaluation set therefore contains file-level labels indicating whether a changed file contains a blocking issue, together with validated line-level annotations and comments when such issues are present. This procedure produces a cleaner and more trustworthy benchmark for offline evaluation than the original raw review data.

After the first evaluation set is constructed, the same general labeling approach can be extended to a larger subset of the raw corpus in order to build a higher-quality training set for the advanced model. In this way, the evaluation pipeline also serves as the basis for creating cleaner supervision for fine-tuning.

## 4. Evaluation

This section describes how we evaluate the quality of the proposed solution. Since the system predicts both the presence of blocking issues and their localization in code, the evaluation includes classification quality, line-level localization quality, and comment quality.

### 4.1 Detection Quality

At the file level, we evaluate whether the model correctly identifies the presence of a blocking issue in a changed Python file.

The main metrics for this part are **Precision**, **Recall**, and **F1-score**. We place special emphasis on precision, since false positive review comments reduce developer trust in the system and create unnecessary review noise. Recall is also important, since the goal of the system is to catch at least part of the blocking issues before human review.

### 4.2 Localization Quality

In addition to detecting an issue, the model must also point to the relevant changed lines. To evaluate this part, we compare predicted line spans with validated ground-truth spans.

We use **Intersection over Union (IoU)** between predicted and reference line ranges as the main localization metric. IoU allows us to measure how closely the predicted span matches the actual problematic region, while still giving partial credit for overlapping predictions. This is important because model predictions and human annotations may differ slightly in exact span boundaries even when they refer to the same issue.

A prediction is considered a correct localization if its overlap with the reference span exceeds a predefined IoU threshold. This allows us to compute localization-aware precision and recall in addition to raw detection metrics.

### 4.3 Comment Quality

To evaluate generated review comments, we compare model outputs with validated human comments for the same issue. Since comments can be phrased in different ways while still conveying the same meaning, we use a semantic similarity metric rather than exact string matching.

Our main automatic metric for comment quality is **BERTScore**. It measures semantic similarity between the generated comment and the reference comment while remaining robust to paraphrasing. We use BERTScore as the primary offline metric for comment evaluation because it is practical, reproducible, and does not require online judge inference for every iteration.

### 4.4 Final Qualitative Evaluation

If time permits, we also plan to perform a final qualitative evaluation using **side-by-side LLM-as-a-judge** assessment. In this setup, a stronger judge model compares the review produced by our system against the human review on the same pull request.

This comparison is performed at the pull request level rather than only at the individual comment level. The judge receives all reviewed changed files in the sampled pull request and compares the quality of line localization and comment usefulness between the two sides. This setup is intended to provide a more realistic measure of practical review quality than isolated file-level metrics alone.

Because this evaluation depends on additional prompt design, judge model selection, and bias mitigation, we treat it as a final optional stage rather than the main offline evaluation protocol.

## 5. Limitations

The proposed system has several important limitations.

First, the current solution is restricted to Python repositories and only processes standard `.py` files. It does not support notebooks, configuration-heavy changes, or multi-language pull requests.

Second, the system reviews pull requests at the file level rather than at the full repository level. This makes the pipeline more practical under limited context budgets, but it also means that some issues requiring broader architectural or cross-file reasoning may be missed.

Third, the quality of the system depends on the quality of retrieved repository context. If relevant local dependencies, usage sites, or repository-specific rules are not retrieved, the model may lack important information needed for correct review.

Fourth, the local context construction step uses truncation to fit large files into the model context. Although this truncation is designed to preserve code most relevant to the change, it may still remove information that would be useful for detecting subtle bugs.

Fifth, the training and evaluation data are derived from human review comments, which are inherently noisy. Some human comments do not correspond to merge-blocking issues, while some real issues may remain uncommented. Although we reduce this noise through evaluation-set validation, the supervision source is still imperfect.

Finally, generated review comments may still contain false positives, incomplete reasoning, or imprecise line localization. Because developer trust is especially sensitive to low-quality review comments, maintaining high precision remains one of the main challenges of the proposed approach.
