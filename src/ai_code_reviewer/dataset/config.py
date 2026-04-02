from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import aiohttp

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

_DATA_ROOT: Path = (
    Path(os.environ["DATA_DIR"]).resolve()
    if os.environ.get("DATA_DIR")
    else PROJECT_ROOT / "data"
)

CHECKPOINT_DIR: Path = _DATA_ROOT / "checkpoints"

DATASET_RAW_PATH: Path = CHECKPOINT_DIR / "dataset_raw.json.gz"
DATASET_FILTERED_PATH: Path = CHECKPOINT_DIR / "filtered.json.gz"
DATASET_FINAL_PATH: Path = CHECKPOINT_DIR / "final.json.gz"

# Dataset download range
RANGE_START = datetime(2022, 1, 3, 0)
RANGE_END = datetime(2022, 1, 3, 0)

# Numbers of snapshot commits to keep
# Should be aligned with GitHub API rate limit
SNAPSHOT_COMMITS_TO_KEEP = 2000

GH_ARCHIVE_API_BASE: str = "https://data.gharchive.org"
GITHUB_API_BASE: str = "https://api.github.com"
GITHUB_GRAPHQL_API: str = "https://api.github.com/graphql"

# Skip blobs larger than this when reading from zipballs (GitHub contenkts API parity).
MAX_FILE_BYTES: int = 10_048_576

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
INCOMING_DEP_MAX_SYMBOLS_PER_FILE: int = 20

GH_ARCHIVE_CONCURRENCY: int = 5
GITHUB_API_CONCURRENCY: int = 5

GITHUB_RATE_LIMIT_MIN_REMAINING: int = 100
GITHUB_RATE_LIMIT_BACKOFF_ENABLED: bool = True
GITHUB_RATE_LIMIT_MAX_SLEEP_SECONDS: float = 120.0

IS_LIKELY_ENGLISH_THRESHOLD: float = 0.7

SEED: int = 42

# Metadata files to search for (matched by basename) anywhere in the repo tree.
METADATA_FILE_NAMES: list[str] = [
    "README.md",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
]

# Sentinel key stored inside the commit-level path_map to hold metadata file content.
# This key must not be a valid repo-relative file path (no path separators, reserved name).
METADATA_FILES_COMMIT_KEY: str = "metadata_files"

# When True, each snapshot commit is augmented with changed .py files that have no
# review comments, capped at the number of commented files in that snapshot.
INCLUDE_NO_COMMENT_FILES: bool = True

HTTP_TIMEOUT_TOTAL: float = 600.0
HTTP_TIMEOUT_CONNECT: float = 30.0
HTTP_TIMEOUT_SOCK_CONNECT: float = 30.0
HTTP_TIMEOUT_SOCK_READ: float = 300.0

HTTP_MAX_RETRY_ATTEMPTS: int = 10
HTTP_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
HTTP_RETRY_AFTER_CAP_SECONDS: float = 120.0
HTTP_EXP_WAIT_MULTIPLIER: float = 1.0
HTTP_EXP_WAIT_MIN_SECONDS: float = 2.0
HTTP_EXP_WAIT_MAX_SECONDS: float = 120.0

# Extra slots above semaphore concurrency for aiohttp connector limit (keep small).
HTTP_CONNECTOR_LIMIT_BUFFER: int = 2


def default_client_timeout() -> aiohttp.ClientTimeout:
    """Return the default `aiohttp` timeout for long downloads and API calls.

    Returns:
        Configured `ClientTimeout` instance.
    """
    return aiohttp.ClientTimeout(
        total=HTTP_TIMEOUT_TOTAL,
        connect=HTTP_TIMEOUT_CONNECT,
        sock_connect=HTTP_TIMEOUT_SOCK_CONNECT,
        sock_read=HTTP_TIMEOUT_SOCK_READ,
    )


def tcp_connector_for_concurrency(concurrency: int) -> aiohttp.TCPConnector:
    """Return a TCP connector sized for at most `concurrency` concurrent requests.

    The limit includes a small buffer so the pool is not tighter than the asyncio
    semaphore used with the same session.

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
