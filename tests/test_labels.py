import pytest

from safety_signal.errors import NotFoundError
from safety_signal.labels import OpenFDALabels, is_single_ingredient, parse_label
from tests.fakes import FakeClient, make_label


def rec(name, **fields):
    return {"set_id": name, "openfda": {"generic_name": [name]}, **fields}


def test_parse_joins_section_lists_and_ignores_other_sections():
    lab = parse_label(rec("METFORMIN", adverse_reactions=["a", "b"], clinical_studies=["skip me"]))
    assert lab.sections == {"adverse_reactions": "a b"}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("METFORMIN HYDROCHLORIDE", True),
        ("SITAGLIPTIN AND METFORMIN HYDROCHLORIDE", False),
        ("METFORMIN, GLIPIZIDE", False),
        ("PIOGLITAZONE/METFORMIN", False),
        ("ASPIRIN", False),
    ],
)
def test_single_ingredient_filter(name, expected):
    assert is_single_ingredient(make_label(generic=name), "metformin") is expected


def test_pagination_stops_on_short_page_and_filters_combinations():
    pages = {0: [rec("METFORMIN"), rec("X AND METFORMIN")], 2: [rec("METFORMIN")]}

    def handler(endpoint, params):
        return {"results": pages[params["skip"]]}

    client = FakeClient(handler)
    out = OpenFDALabels(client, page_size=2, max_records=10).labels("metformin")
    assert [c[1]["skip"] for c in client.calls] == [0, 2]
    assert len(out) == 2


def test_max_records_caps_the_fetch():
    client = FakeClient(lambda e, p: {"results": [rec("METFORMIN")] * p["limit"]})
    out = OpenFDALabels(client, page_size=5, max_records=7).labels("metformin")
    assert len(out) == 7


def test_no_matches_gives_empty_list():
    def handler(endpoint, params):
        raise NotFoundError()

    assert OpenFDALabels(FakeClient(handler)).labels("nothing") == []
