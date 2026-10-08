"""Thin openFDA HTTP client: cache, rate cap, retries. Knows nothing about drugs or reactions."""
import time
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import httpx

from safety_signal.cache import NullCache
from safety_signal.errors import ApiError, NotFoundError, RateLimitError
from safety_signal.ratelimit import RateLimiter

BASE_URL = "https://api.fda.gov"


class OpenFDAClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        http: Any = None,
        cache: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
        max_per_minute: int = 200,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
        base_url: str = BASE_URL,
    ):
        self._api_key = api_key
        self._http = http if http is not None else httpx.Client(timeout=30.0)
        self._cache = cache if cache is not None else NullCache()
        self._sleep = sleep
        self._limiter = RateLimiter(max_per_minute, now=now, sleep=sleep)
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._base_url = base_url.rstrip("/")

    @staticmethod
    def cache_key(endpoint: str, params: dict) -> str:
        # The API key is deliberately not part of the key or anything stored.
        return f"{endpoint}?{urlencode(sorted((k, str(v)) for k, v in params.items()))}"

    def get(self, endpoint: str, params: dict) -> dict:
        key = self.cache_key(endpoint, params)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        body = self._fetch(endpoint, params)
        self._cache.put(key, body)
        return body

    def _fetch(self, endpoint: str, params: dict) -> dict:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        sent = dict(params)
        if self._api_key:
            sent["api_key"] = self._api_key
        for attempt in range(self._max_retries + 1):
            self._limiter.wait()
            try:
                resp = self._http.get(url, params=sent)
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    self._sleep(self._backoff * 2**attempt)
                    continue
                raise ApiError(0, f"network error: {type(exc).__name__}") from None
            status = resp.status_code
            if status == 200:
                return resp.json()
            if status == 404:
                raise NotFoundError()
            if status == 429 or status >= 500:
                if attempt < self._max_retries:
                    self._sleep(self._backoff * 2**attempt)
                    continue
                if status == 429:
                    raise RateLimitError(status, "still rate limited after retries")
                raise ApiError(status, "server error after retries")
            raise ApiError(status, resp.text[:200])
        raise AssertionError("unreachable")
