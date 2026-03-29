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
DATASET_ENRICHED_PATH: Path = CHECKPOINT_DIR / "enriched.json.gz"
DATASET_FINAL_PATH: Path = CHECKPOINT_DIR / "final.json.gz"

# Dataset download range
RANGE_START = datetime(2021, 1, 1, 0)
RANGE_END = datetime(2021, 1, 2, 0)

# Numbers of snapshot commits to keep
# Should be aligned with GitHub API rate limit
SNAPSHOT_COMMITS_TO_KEEP = 1000

GH_ARCHIVE_API_BASE: str = "https://data.gharchive.org"
GITHUB_API_BASE: str = "https://api.github.com"

# Skip blobs larger than this when reading from zipballs (GitHub contenkts API parity).
MAX_FILE_BYTES: int = 10_048_576

GH_ARCHIVE_CONCURRENCY: int = 5
GITHUB_API_CONCURRENCY: int = 5

GITHUB_RATE_LIMIT_MIN_REMAINING: int = 100
GITHUB_RATE_LIMIT_BACKOFF_ENABLED: bool = True
GITHUB_RATE_LIMIT_MAX_SLEEP_SECONDS: float = 120.0

IS_LIKELY_ENGLISH_THRESHOLD: float = 0.7

DATASET_SCHEMA_VERSION: int = 1

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
    """Return the default ``aiohttp`` timeout for long downloads and API calls.

    Returns:
        Configured ``ClientTimeout`` instance.
    """
    return aiohttp.ClientTimeout(
        total=HTTP_TIMEOUT_TOTAL,
        connect=HTTP_TIMEOUT_CONNECT,
        sock_connect=HTTP_TIMEOUT_SOCK_CONNECT,
        sock_read=HTTP_TIMEOUT_SOCK_READ,
    )


def tcp_connector_for_concurrency(concurrency: int) -> aiohttp.TCPConnector:
    """Return a TCP connector sized for at most ``concurrency`` concurrent requests.

    The limit includes a small buffer so the pool is not tighter than the asyncio
    semaphore used with the same session.

    Args:
        concurrency:
            Expected maximum simultaneous in-flight HTTP calls (e.g. semaphore value).

    Returns:
        Configured ``TCPConnector`` for ``ClientSession(connector=...)``.
    """
    limit = max(concurrency + HTTP_CONNECTOR_LIMIT_BUFFER, 4)
    return aiohttp.TCPConnector(limit=limit)


def get_github_token() -> str:
    """Return the GitHub token from the environment (empty string if unset).

    Returns:
        Value of ``GITHUB_TOKEN``, or ``""``.
    """
    return os.environ.get("GITHUB_TOKEN", "")


def github_api_headers() -> dict[str, str]:
    """Build GitHub REST API headers for the current process environment.

    Call after ``load_dotenv()`` so ``GITHUB_TOKEN`` is visible.

    Returns:
        Headers including ``Accept`` and ``Authorization`` when a token is set.
    """
    token = get_github_token()
    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    return headers
