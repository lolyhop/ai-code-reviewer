"""Fallback mock data for the demo.

Used **only** when the user explicitly enables the "Use mock data" toggle in
the UI, or when live GitHub access fails and the user opts into mock mode.

The default UI path is real PR analysis (see ``demo.adapters``); mock data
exists for offline rehearsals, screen recordings, and unit-style smoke tests.
"""

from __future__ import annotations

import typing as tp


MOCK_PR_URL = "https://github.com/example-org/payment-service/pull/142"


MOCK_PR_META: tp.Dict[str, tp.Any] = {
    "repo": "example-org/payment-service",
    "pr_number": 142,
    "pr_title": "Fix token refresh logic",
    "pr_body": (
        "Refactors the token refresh helper to cache generated tokens per user.\n\n"
        "- Avoids redundant calls to `generate_new_token` when the token is still valid.\n"
        "- Adds early return for cached entries."
    ),
    "base_sha": "f4a1c0e",
    "head_sha": "9b3d712",
    "html_url": MOCK_PR_URL,
    "repo_star_count": 2340,
}


MOCK_FILES: tp.List[tp.Dict[str, tp.Any]] = [
    {
        "path": "src/auth/tokens.py",
        "language": "python",
        "status": "blocking_issue",
        "additions": 6,
        "deletions": 1,
        "diff": (
            "@@ -12,7 +12,12 @@\n"
            " from src.auth.utils import generate_new_token\n"
            " from src.auth.validators import validate_user_id\n"
            " \n"
            "-def refresh_token(user_id):\n"
            "+def refresh_token(user_id, cache={}):\n"
            "+    if user_id in cache:\n"
            "+        return cache[user_id]\n"
            "+    token = generate_new_token(user_id)\n"
            "+    cache[user_id] = token\n"
            "+    return token\n"
        ),
        "source_lines": [
            (12, " from src.auth.utils import generate_new_token"),
            (13, " from src.auth.validators import validate_user_id"),
            (14, " "),
            (15, "+def refresh_token(user_id, cache={}):"),
            (16, "+    if user_id in cache:"),
            (17, "+        return cache[user_id]"),
            (18, "+    token = generate_new_token(user_id)"),
            (19, "+    cache[user_id] = token"),
            (20, "+    return token"),
        ],
        "issues": [
            {
                "type": "Blocking issue: default mutable argument",
                "line_start": 15,
                "line_end": 15,
                "comment": (
                    "`cache={}` is a mutable default argument. The dict is shared across "
                    "all calls to `refresh_token`, so cached tokens persist between "
                    "unrelated invocations and across different request contexts. This "
                    "can leak tokens between users in a multi-tenant environment."
                ),
                "suggestion": (
                    "def refresh_token(user_id, cache=None):\n"
                    "    if cache is None:\n"
                    "        cache = {}"
                ),
                "confidence": 0.94,
            },
        ],
        "error": None,
    },
    {
        "path": "src/api/users.py",
        "language": "python",
        "status": "clean",
        "additions": 2,
        "deletions": 0,
        "diff": (
            "@@ -45,6 +45,8 @@\n"
            " class UserService:\n"
            "     def __init__(self, db: Database):\n"
            "         self.db = db\n"
            "+        self.logger = logging.getLogger(__name__)\n"
            "+        self.logger.info('UserService initialized')\n"
        ),
        "source_lines": [
            (45, " class UserService:"),
            (46, "     def __init__(self, db: Database):"),
            (47, "         self.db = db"),
            (48, "+        self.logger = logging.getLogger(__name__)"),
            (49, "+        self.logger.info('UserService initialized')"),
        ],
        "issues": [],
        "error": None,
    },
    {
        "path": "src/api/payments.py",
        "language": "python",
        "status": "clean",
        "additions": 1,
        "deletions": 1,
        "diff": (
            "@@ -78,7 +78,7 @@\n"
            "     def process_payment(self, amount: float) -> bool:\n"
            "-        return self._gateway.charge(amount)\n"
            "+        return self._gateway.charge(round(amount, 2))\n"
        ),
        "source_lines": [
            (78, "     def process_payment(self, amount: float) -> bool:"),
            (79, "+        return self._gateway.charge(round(amount, 2))"),
        ],
        "issues": [],
        "error": None,
    },
    {
        "path": "notebooks/eda.ipynb",
        "language": "notebook",
        "status": "skipped",
        "additions": 14,
        "deletions": 3,
        "diff": "",
        "source_lines": [],
        "issues": [],
        "error": None,
    },
]


MOCK_CONTEXT: tp.Dict[str, tp.Any] = {
    "pr_title": MOCK_PR_META["pr_title"],
    "pr_body": MOCK_PR_META["pr_body"],
    "changed_file_context": (
        "src/auth/tokens.py — token refresh utility (12 → 17 lines after patch)"
    ),
    "imported_definitions": [
        "generate_new_token(user_id: str) -> str   [src/auth/utils.py:34]",
        "validate_user_id(uid: str) -> bool         [src/auth/validators.py:12]",
    ],
    "usage_sites": [
        "src/api/auth_router.py:87       calls refresh_token(request.user.id)",
        "src/workers/session_cleanup.py:23  calls refresh_token(uid)",
    ],
    "repo_metadata": (
        "Repository: example-org/payment-service\n"
        "Stars: 2,340 | Language: Python | License: MIT\n"
        "Default branch: main | CI: GitHub Actions"
    ),
}


def build_mock_pr_review() -> tp.Dict[str, tp.Any]:
    """Return a mock ``PRReview``-shaped dict (same layout as adapters output)."""
    return {
        "meta": dict(MOCK_PR_META),
        "files": [dict(f) for f in MOCK_FILES],
        "context": dict(MOCK_CONTEXT),
        "is_mock": True,
        "model_info": {
            "name": "mock://demo-fixture",
            "device": "n/a",
            "loaded": False,
        },
    }
