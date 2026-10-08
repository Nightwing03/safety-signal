from safety_signal.cache import MemoryCache, NullCache, SqliteCache


def test_null_cache_never_hits():
    c = NullCache()
    c.put("k", {"a": 1})
    assert c.get("k") is None


def test_memory_cache_round_trip():
    c = MemoryCache()
    c.put("k", {"a": 1})
    assert c.get("k") == {"a": 1}
    assert c.get("missing") is None


def test_sqlite_round_trip_and_persistence(tmp_path):
    path = str(tmp_path / "nested" / "c.sqlite")
    first = SqliteCache(path)
    first.put("k", {"a": [1, 2]})
    first.close()
    second = SqliteCache(path)
    assert second.get("k") == {"a": [1, 2]}
    second.close()


def test_sqlite_entries_expire(tmp_path):
    now = {"t": 1000.0}
    c = SqliteCache(str(tmp_path / "c.sqlite"), ttl_seconds=100, time_fn=lambda: now["t"])
    c.put("k", {"a": 1})
    now["t"] = 1099.0
    assert c.get("k") == {"a": 1}
    now["t"] = 1101.0
    assert c.get("k") is None


def test_sqlite_put_overwrites(tmp_path):
    c = SqliteCache(str(tmp_path / "c.sqlite"))
    c.put("k", {"v": 1})
    c.put("k", {"v": 2})
    assert c.get("k") == {"v": 2}


def test_sqlite_cache_can_be_used_from_other_threads(tmp_path):
    import threading

    c = SqliteCache(str(tmp_path / "c.sqlite"))
    errors = []

    def work(i):
        try:
            c.put(f"k{i}", {"v": i})
            assert c.get(f"k{i}") == {"v": i}
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
