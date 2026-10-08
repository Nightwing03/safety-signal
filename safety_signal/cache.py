"""Response caches. All share two methods: get(key) -> dict | None and put(key, value)."""
import json
import sqlite3
import time
from pathlib import Path
from typing import Callable, Optional


class NullCache:
    def get(self, key: str) -> Optional[dict]:
        return None

    def put(self, key: str, value: dict) -> None:
        pass


class MemoryCache:
    def __init__(self):
        self.data: dict[str, dict] = {}

    def get(self, key: str) -> Optional[dict]:
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value


class SqliteCache:
    """Disk cache with a time-to-live. Expired rows are ignored on read and overwritten on write."""

    def __init__(self, path: str, ttl_seconds: float = 86400.0, time_fn: Callable[[], float] = time.time):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._time = time_fn
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, body TEXT NOT NULL, fetched_at REAL NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str) -> Optional[dict]:
        row = self._db.execute("SELECT body, fetched_at FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        body, fetched_at = row
        if self._time() - fetched_at > self._ttl:
            return None
        return json.loads(body)

    def put(self, key: str, value: dict) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO cache (key, body, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), self._time()),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
