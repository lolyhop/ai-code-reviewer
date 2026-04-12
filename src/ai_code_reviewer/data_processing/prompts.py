NO_COMMENT_VALIDATION_PROMPT = """You are an expert Python code reviewer acting as a quality-assurance judge.

You are given a file that was changed in a pull request but received NO human review comments.
Decide whether this file contains a blocking issue.

A blocking issue is a defect that can break correctness, reliability, security, or expected production behavior.
Style preferences, formatting, naming, documentation, missing tests, and optional refactorings are NOT blocking issues.

Context
=======

Repository:
{repository_metadata}

Pull request:
{pull_request_metadata}

Project structure (top 2 levels):
{file_tree}

Changed file:
{patched_content}

Files imported by the changed file (outgoing dependencies):
{outgoing_dependencies}

Files that import from the changed file (incoming dependencies):
{incoming_dependencies}

Answer with a single word: true if the file contains a blocking issue, false otherwise.
""".strip()


VALID_CATEGORIES = frozenset(
    {
        "blocking_issue",
        "style",
        "performance",
        "best_practice",
        "documentation",
        "question",
        "nitpick",
        "praise",
        "other",
    }
)


COMMENT_CLASSIFICATION_PROMPT = """You are an expert at analyzing code-review feedback.

You are given a changed file in a pull request together with human review comments.
Classify each comment into exactly one category.

Categories
----------
- blocking_issue  — correctness bugs, security vulnerabilities, data-loss risks, crashes, race conditions
- style           — formatting, naming conventions, import ordering, whitespace
- performance     — efficiency concerns, unnecessary allocations, algorithmic complexity
- best_practice   — design patterns, idiomatic code, architectural suggestions, error-handling improvements
- documentation   — missing / incorrect docstrings, comments, type hints
- question        — reviewer asking for clarification or explanation
- nitpick         — minor optional observations that don't affect correctness or style
- praise          — positive feedback, approval, compliments
- other           — does not fit any category above

Context
=======

Repository:
{repository_metadata}

Pull request:
{pull_request_metadata}

Project structure (top 2 levels):
{file_tree}

Changed file:
{patched_content}

Files imported by the changed file (outgoing dependencies):
{outgoing_dependencies}

Files that import from the changed file (incoming dependencies):
{incoming_dependencies}

Review comments to classify:
{comments}

Answer with one line per comment in the format:
<comment_index>: <category>

Example:
0: blocking_issue
1: style
2: question
""".strip()
