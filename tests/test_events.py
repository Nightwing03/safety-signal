import pytest

from safety_signal.errors import DataError, NotFoundError
from safety_signal.events import OpenFDAEvents
from tests.fakes import FakeClient, total_body


def test_overall_total_uses_limit_one_and_no_search():
    client = FakeClient(lambda e, p: total_body(12345))
    assert OpenFDAEvents(client).total() == 12345
    _, params = client.calls[0]
    assert params == {"limit": 1}


def test_drug_total_searches_the_drug():
    client = FakeClient(lambda e, p: total_body(7))
    assert OpenFDAEvents(client).drug_total("metformin") == 7
    assert "metformin" in client.calls[0][1]["search"]


def test_pair_search_contains_both_terms_joined_by_and():
    client = FakeClient(lambda e, p: total_body(3))
    OpenFDAEvents(client).pair_count("metformin", "NAUSEA")
    search = client.calls[0][1]["search"]
    assert "metformin" in search and "NAUSEA" in search and " AND " in search


def test_no_matches_means_zero():
    def handler(endpoint, params):
        raise NotFoundError()

    events = OpenFDAEvents(FakeClient(handler))
    assert events.pair_count("x", "y") == 0
    assert events.top_reactions("x") == []


def test_top_reactions_returns_terms_in_order():
    body = {"meta": {"last_updated": "2026-01-01"}, "results": [{"term": "A", "count": 9}, {"term": "B", "count": 3}]}
    events = OpenFDAEvents(FakeClient(lambda e, p: body))
    assert events.top_reactions("x") == ["A", "B"]


def test_as_of_comes_from_response_metadata():
    events = OpenFDAEvents(FakeClient(lambda e, p: total_body(1, last_updated="2030-05-06")))
    assert events.as_of() is None
    events.total()
    assert events.as_of() == "2030-05-06"


def test_missing_meta_or_total_raises_data_error():
    with pytest.raises(DataError):
        OpenFDAEvents(FakeClient(lambda e, p: {"results": []})).total()
    with pytest.raises(DataError):
        OpenFDAEvents(FakeClient(lambda e, p: {"meta": {"results": {}}})).total()


@pytest.mark.parametrize("bad", ['a"b', "", "   "])
def test_unsafe_search_terms_are_rejected_before_any_request(bad):
    client = FakeClient(lambda e, p: total_body(1))
    with pytest.raises(ValueError):
        OpenFDAEvents(client).drug_total(bad)
    assert client.calls == []
