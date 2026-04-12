from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

DATA_ROOT: Path = Path(os.environ["DATA_DIR"]).resolve() if os.environ.get("DATA_DIR") else PROJECT_ROOT / "data"

CHECKPOINT_DIR: Path = DATA_ROOT / "checkpoints"
DATASET_CHECKPOINT_RAW_PATH: Path = CHECKPOINT_DIR / "checkpoint_raw.json.zst"
DATASET_CHECKPOINT_PR_ENRICHED_PATH: Path = CHECKPOINT_DIR / "checkpoint_pr_enriched.json.zst"
DATASET_CHECKPOINT_FINAL_PATH: Path = CHECKPOINT_DIR / "checkpoint_final.json.zst"

EXPORTS_DIR: Path = DATA_ROOT / "exports"
DATASET_PATH: Path = EXPORTS_DIR / "dataset.parquet"

# Dataset download range
RANGE_START = datetime(2022, 1, 1, 0, tzinfo=timezone.utc)
RANGE_END = datetime(2022, 1, 10, 0, tzinfo=timezone.utc)

# Numbers of snapshot commits to keep
# Should be aligned with GitHub API rate limit
SNAPSHOT_COMMITS_TO_KEEP = 2000

GH_ARCHIVE_API_BASE: str = "https://data.gharchive.org"
GITHUB_API_BASE: str = "https://api.github.com"
GITHUB_GRAPHQL_API: str = "https://api.github.com/graphql"

# Skip blobs larger than this when reading from zipballs (GitHub contents API parity).
FILE_MAX_BYTES: int = 2 * 1024 * 1024
ZIPBALL_MAX_BYTES: int = 50 * 1024 * 1024

# Source-root prefixes tried when resolving absolute Python imports.
# The empty string represents the repository root itself; "src" covers the
# common `src/` layout used by setuptools/poetry projects.  Both
# `import_resolution.resolve_import_candidates` (forward direction: import
# statement → candidate paths) and `github_api._incoming_import_target_variants`
# (inverse direction: changed file path → expected import candidates) use this
# as their single source of truth.
IMPORT_SOURCE_ROOTS: tuple[str, ...] = ("", "src")

# Minimum character length for a symbol name to be included in incoming-dependency
# search.  Short names (e.g. `do`, `ok`) match too broadly across a repo.
INCOMING_DEP_MIN_SYMBOL_LENGTH: int = 3

# Maximum number of symbols (function/class/method names) collected per changed
# file for incoming-dependency search.  When a file has more qualifying symbols
# the longest names are kept (more specific identifiers produce fewer false
# positives).  Every changed file always participates — only its symbol set is
# trimmed, never the file itself.
INCOMING_DEP_MAX_SYMBOLS_PER_FILE: int = 30

# Maximum recursion depth when resolving package re-export chains for outgoing
# dependencies (for example `pkg/__init__.py -> pkg/sub/__init__.py -> pkg/mod.py`).
OUTGOING_REEXPORT_MAX_DEPTH: int = 4

GH_ARCHIVE_CONCURRENCY: int = 10
GITHUB_API_CONCURRENCY: int = 10

GITHUB_RATE_LIMIT_MIN_REMAINING: int = 100
GITHUB_RATE_LIMIT_BACKOFF_ENABLED: bool = True
GITHUB_RATE_LIMIT_MAX_SLEEP_SECONDS: float = 120.0

IS_LIKELY_ENGLISH_THRESHOLD: float = 0.7

SEED: int = 42

# Metadata files to search for (matched by basename) anywhere in the repo tree.
METADATA_FILE_NAMES: list[str] = [
    "README.md",
    "CONTRIBUTING.md",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
]

# When True, each snapshot commit is augmented with changed .py files that have no
# review comments, capped at the number of commented files in that snapshot.
INCLUDE_NO_COMMENT_FILES: bool = True

HTTP_JSON_TIMEOUT_TOTAL: float = 20.0
HTTP_JSON_TIMEOUT_CONNECT: float = 20.0
HTTP_JSON_TIMEOUT_SOCK_CONNECT: float = 20.0
HTTP_JSON_TIMEOUT_SOCK_READ: float = 20.0

# Timeout profile for zipball downloads and large archive reads.
HTTP_ZIP_TIMEOUT_TOTAL: float = 120.0
HTTP_ZIP_TIMEOUT_CONNECT: float = 30.0
HTTP_ZIP_TIMEOUT_SOCK_CONNECT: float = 30.0
HTTP_ZIP_TIMEOUT_SOCK_READ: float = 120.0

HTTP_MAX_RETRY_ATTEMPTS: int = 1
HTTP_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
HTTP_RETRY_AFTER_CAP_SECONDS: float = 120.0
HTTP_EXP_WAIT_MULTIPLIER: float = 1.0
HTTP_EXP_WAIT_MIN_SECONDS: float = 2.0
HTTP_EXP_WAIT_MAX_SECONDS: float = 30.0

# Extra slots above semaphore concurrency for aiohttp connector limit (keep small).
HTTP_CONNECTOR_LIMIT_BUFFER: int = 2


def tcp_connector_for_concurrency(concurrency: int) -> aiohttp.TCPConnector:
    """Return a TCP connector sized for at most `concurrency` concurrent requests.

    Args:
        concurrency:
            Expected maximum simultaneous in-flight HTTP calls (e.g. semaphore value).

    Returns:
        Configured `TCPConnector` for `ClientSession(connector=...)`.
    """
    limit = max(concurrency + HTTP_CONNECTOR_LIMIT_BUFFER, 4)
    return aiohttp.TCPConnector(limit=limit)


def get_github_token() -> str:
    """Return the GitHub token from the environment (empty string if unset).

    Returns:
        Value of `GITHUB_TOKEN`, or `""`.
    """
    return os.environ.get("GITHUB_TOKEN", "")


def github_api_headers() -> dict[str, str]:
    """Build GitHub REST API headers for the current process environment.

    Returns:
        Headers including `Accept` and `Authorization` when a token is set.
    """
    token = get_github_token()
    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def github_graphql_headers() -> dict[str, str]:
    """Build GitHub GraphQL API headers for the current process environment.

    Returns:
        Headers including `Content-Type` and `Authorization` when a token
        is set.
    """
    token = get_github_token()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
