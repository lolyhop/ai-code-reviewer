from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
)
from tenacity.wait import wait_random_exponential

from ai_code_reviewer.dataset import config

logger = logging.getLogger(__name__)

_EXP_WAIT = wait_random_exponential(
    multiplier=config.HTTP_EXP_WAIT_MULTIPLIER,
    min=config.HTTP_EXP_WAIT_MIN_SECONDS,
    max=config.HTTP_EXP_WAIT_MAX_SECONDS,
)


class RetryableHTTPStatusError(Exception):
    """Raised when the server returns a status that should be retried."""

    def __init__(self, status: int, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def _parse_retry_after(response: aiohttp.ClientResponse) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _wait_retry_after_or_exponential(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception()
    if isinstance(exc, RetryableHTTPStatusError) and exc.retry_after is not None:
        return min(float(exc.retry_after), config.HTTP_RETRY_AFTER_CAP_SECONDS)
    return _EXP_WAIT(retry_state)


def _should_retry_http(exc: BaseException) -> bool:
    if isinstance(exc, RetryableHTTPStatusError):
        return True
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, aiohttp.ClientError):
        return True
    return False


def _is_github_rest_url(url: str) -> bool:
    return "api.github.com" in url


def _github_rate_limit_sleep_seconds(
    url: str,
    status: int,
    response_headers: aiohttp.typedefs.LooseHeaders,
) -> float:
    """Return seconds to sleep after a response when GitHub quota is low.

    Does not apply to retryable error statuses (handled by tenacity). Proactive
    backoff runs only after releasing the semaphore.

    Args:
        url:
            Request URL.
        status:
            HTTP status of the completed response.
        response_headers:
            Response headers (while or after response is read).

    Returns:
        Non-negative sleep duration in seconds, capped by config.
    """
    if not config.GITHUB_RATE_LIMIT_BACKOFF_ENABLED:
        return 0.0
    if status != 200:
        return 0.0
    if not _is_github_rest_url(url):
        return 0.0
    rem_raw = response_headers.get("X-RateLimit-Remaining")
    reset_raw = response_headers.get("X-RateLimit-Reset")
    if rem_raw is None or reset_raw is None:
        return 0.0
    try:
        remaining = int(str(rem_raw))
        reset_ts = float(str(reset_raw))
    except (TypeError, ValueError):
        return 0.0
    if remaining > config.GITHUB_RATE_LIMIT_MIN_REMAINING:
        return 0.0
    sleep_for = max(0.0, reset_ts - time.time())
    return min(sleep_for, config.GITHUB_RATE_LIMIT_MAX_SLEEP_SECONDS)


async def _maybe_github_rate_limit_sleep(
    url: str,
    status: int,
    response_headers: aiohttp.typedefs.LooseHeaders,
) -> None:
    delay = _github_rate_limit_sleep_seconds(url, status, response_headers)
    if delay > 0:
        logger.debug(
            "GitHub rate-limit proactive backoff: sleeping %.2fs (url=%s)",
            delay,
            url[:80],
        )
        await asyncio.sleep(delay)


async def async_http_get_bytes(
    session: aiohttp.ClientSession,
    url: str,
    *,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """GET ``url`` and return ``(status, body)``.

    Retries on transient HTTP statuses, timeouts, and connection errors.
    Does not retry on 404 or other final client errors (caller handles status).

    Args:
        session:
            Shared aiohttp session (with timeout configured).
        url:
            Full URL.
        semaphore:
            Concurrency limiter acquired per attempt.
        headers:
            Optional request headers.

    Returns:
        Response status and raw body bytes.
    """

    async def _attempt() -> tuple[int, bytes]:
        status: int = 0
        body: bytes = b""
        async with semaphore:
            async with session.get(url, headers=headers) as response:
                status = response.status
                if status in config.HTTP_RETRY_STATUSES:
                    ra = _parse_retry_after(response)
                    await response.read()
                    raise RetryableHTTPStatusError(status, retry_after=ra)
                body = await response.read()
                hdrs = response.headers
        await _maybe_github_rate_limit_sleep(url, status, hdrs)
        return status, body

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(config.HTTP_MAX_RETRY_ATTEMPTS),
        wait=_wait_retry_after_or_exponential,
        retry=retry_if_exception(_should_retry_http),
        before_sleep=lambda retry_state: logger.debug(
            "HTTP retry %s after %s",
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        ),
        reraise=True,
    ):
        with attempt:
            return await _attempt()
    raise RuntimeError("async_http_get_bytes: unreachable")


async def async_http_get_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    semaphore: asyncio.Semaphore,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any | None]:
    """GET ``url`` and parse JSON on HTTP 200.

    Retries on transient failures. On non-200 responses returns ``(status, None)`` without
    parsing (no retry for 404/401/403 unless they are listed as transient — they are not).

    Args:
        session:
            Shared aiohttp session.
        url:
            Full URL.
        semaphore:
            Concurrency limiter.
        headers:
            Optional request headers.

    Returns:
        ``(status, parsed_json)`` where ``parsed_json`` is only set for status 200.
    """

    async def _attempt() -> tuple[int, Any | None]:
        status: int = 0
        parsed: Any | None = None
        hdrs: aiohttp.typedefs.LooseHeaders | None = None
        async with semaphore:
            async with session.get(url, headers=headers) as response:
                status = response.status
                text = await response.text()
                if status in config.HTTP_RETRY_STATUSES:
                    ra = _parse_retry_after(response)
                    raise RetryableHTTPStatusError(status, retry_after=ra)
                hdrs = response.headers
                if status == 200:
                    parsed = json.loads(text)
        if hdrs is not None:
            await _maybe_github_rate_limit_sleep(url, status, hdrs)
        if status != 200:
            return status, None
        return status, parsed

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(config.HTTP_MAX_RETRY_ATTEMPTS),
        wait=_wait_retry_after_or_exponential,
        retry=retry_if_exception(_should_retry_http),
        before_sleep=lambda retry_state: logger.debug(
            "HTTP JSON retry %s after %s",
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        ),
        reraise=True,
    ):
        with attempt:
            return await _attempt()
    raise RuntimeError("async_http_get_json: unreachable")
