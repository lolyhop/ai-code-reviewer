from __future__ import annotations

import asyncio
import json
import logging
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
        async with semaphore:
            async with session.get(url, headers=headers) as response:
                status = response.status
                if status in config.HTTP_RETRY_STATUSES:
                    ra = _parse_retry_after(response)
                    await response.read()
                    raise RetryableHTTPStatusError(status, retry_after=ra)
                body = await response.read()
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
        async with semaphore:
            async with session.get(url, headers=headers) as response:
                status = response.status
                text = await response.text()
                if status in config.HTTP_RETRY_STATUSES:
                    ra = _parse_retry_after(response)
                    raise RetryableHTTPStatusError(status, retry_after=ra)
                if status != 200:
                    return status, None
                return status, json.loads(text)

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
