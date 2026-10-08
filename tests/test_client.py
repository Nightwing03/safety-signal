import httpx
import pytest

from safety_signal.cache import MemoryCache
from safety_signal.client import OpenFDAClient
from safety_signal.errors import ApiError, NotFoundError, RateLimitError
from tests.fakes import FakeClock, FakeHttp, FakeResponse

OK = {"meta": {}, "results": []}


def make(responses, api_key="SECRET", **kw):
    clock = FakeClock()
    http = FakeHttp(responses, clock=clock)
    cache = MemoryCache()
    client = OpenFDAClient(
        api_key=api_key, http=http, cache=cache, sleep=clock.sleep, now=clock.now, **kw
    )
    return client, http, cache, clock


def test_identical_requests_hit_the_network_once():
    client, http, _, _ = make([FakeResponse(200, OK)])
    assert client.get("drug/event.json", {"limit": 1}) == OK
    assert client.get("drug/event.json", {"limit": 1}) == OK
    assert len(http.calls) == 1


def test_api_key_is_sent_but_never_cached():
    client, http, cache, _ = make([FakeResponse(200, OK)])
    client.get("drug/event.json", {"limit": 1})
    assert http.calls[0][1]["api_key"] == "SECRET"
    assert all("SECRET" not in key for key in cache.data)


def test_no_key_means_no_key_param():
    client, http, _, _ = make([FakeResponse(200, OK)], api_key=None)
    client.get("drug/event.json", {"limit": 1})
    assert "api_key" not in http.calls[0][1]


def test_404_raises_not_found_and_is_not_cached():
    client, _, cache, _ = make([FakeResponse(404)])
    with pytest.raises(NotFoundError):
        client.get("drug/event.json", {"search": "x"})
    assert cache.data == {}


def test_429_is_retried_with_backoff_then_succeeds():
    client, http, _, clock = make([FakeResponse(429), FakeResponse(200, OK)])
    assert client.get("drug/event.json", {"limit": 1}) == OK
    assert len(http.calls) == 2
    assert clock.sleeps and all(s > 0 for s in clock.sleeps)


def test_persistent_429_raises_rate_limit_error_after_all_retries():
    client, http, _, _ = make([FakeResponse(429)] * 4, max_retries=3)
    with pytest.raises(RateLimitError):
        client.get("drug/event.json", {"limit": 1})
    assert len(http.calls) == 4


def test_persistent_500_raises_api_error():
    client, http, _, _ = make([FakeResponse(500)] * 4, max_retries=3)
    with pytest.raises(ApiError) as info:
        client.get("drug/event.json", {"limit": 1})
    assert info.value.status == 500
    assert len(http.calls) == 4


def test_client_errors_are_not_retried_and_do_not_leak_the_key():
    client, http, _, _ = make([FakeResponse(400, text="bad query")])
    with pytest.raises(ApiError) as info:
        client.get("drug/event.json", {"limit": 1})
    assert len(http.calls) == 1
    assert "SECRET" not in str(info.value)


def test_transport_error_is_retried():
    client, http, _, _ = make([httpx.ConnectError("boom"), FakeResponse(200, OK)])
    assert client.get("drug/event.json", {"limit": 1}) == OK
    assert len(http.calls) == 2


def test_request_rate_is_capped():
    responses = [FakeResponse(200, OK) for _ in range(8)]
    client, http, _, _ = make(responses, max_per_minute=3)
    for i in range(8):
        client.get("drug/event.json", {"limit": i + 1})
    for i in range(len(http.times) - 3):
        assert http.times[i + 3] - http.times[i] >= 60.0
