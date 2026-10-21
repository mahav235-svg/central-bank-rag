"""A caching, rate-limited HTTP fetcher.

Every response body is written to ``raw_dir`` keyed by a hash of the request, so a re-run of the
ingest touches the network only for documents it has never seen. That keeps us polite to the two
central banks' servers and makes ingestion reproducible offline once the cache is warm.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


class Fetcher:
    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        min_interval: float = 1.0,
        retries: int = 3,
        timeout: float = 60.0,
        offline: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self.offline = offline
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._last_call: dict[str, float] = {}

    def _cache_file(self, method: str, url: str, data: dict[str, str] | None) -> Path:
        key = json.dumps([method, url, sorted((data or {}).items())], ensure_ascii=False)
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        host = urlparse(url).netloc.replace(":", "_")
        return self.cache_dir / host / f"{digest}.bin"

    def get(self, url: str, cache: bool = True) -> bytes:
        """Fetch a URL. Pass ``cache=False`` for index pages that change as new documents appear."""
        return self._request("GET", url, None, cache=cache)

    def post(self, url: str, data: dict[str, str], cache: bool = True) -> bytes:
        return self._request("POST", url, data, cache=cache)

    def _request(self, method: str, url: str, data: dict[str, str] | None, cache: bool = True) -> bytes:
        """``cache=False`` means "prefer a fresh copy": it still stores the response, so an offline
        run can fall back to the last copy seen."""
        path = self._cache_file(method, url, data)
        if path.exists() and (cache or self.offline):
            return path.read_bytes()
        if self.offline:
            raise FetchError(f"offline and not cached: {url}")
        body = self._send(method, url, data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body

    def _send(self, method: str, url: str, data: dict[str, str] | None) -> bytes:
        host = urlparse(url).netloc
        backoff = 2.0
        for attempt in range(1, self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last_call[host] = time.monotonic()
            try:
                resp = self._session.request(method, url, data=data, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("%s %s failed on attempt %d: %s", method, url, attempt, exc)
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 404:
                    raise FetchError(f"404 for {url}")
                log.warning("%s %s returned %d on attempt %d", method, url, resp.status_code, attempt)
            time.sleep(backoff)
            backoff *= 2
        raise FetchError(f"{method} {url} failed after {self.retries} attempts")
