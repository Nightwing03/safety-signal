import pytest

from safety_signal.errors import DataError
from safety_signal.pairs import analyze_drug, build_table
from tests.fakes import make_events


@pytest.mark.parametrize(
    "n, drug, reaction, pair",
    [(1000, 100, 200, 50), (1000, 100, 100, 100), (50, 10, 5, 0), (10**6, 5000, 20000, 400)],
)
def test_build_table_cells_add_up(n, drug, reaction, pair):
    t = build_table(n, drug, reaction, pair)
    assert t.n == n
    assert t.a + t.b == drug
    assert t.a + t.c == reaction


@pytest.mark.parametrize("args", [(100, 10, 50, 20), (100, 10, 5, 8), (100, 60, 60, 10)])
def test_inconsistent_counts_raise_data_error(args):
    with pytest.raises(DataError):
        build_table(*args)


def test_every_result_table_matches_the_source_counts():
    ev = make_events()
    for r in analyze_drug(ev, "alpha"):
        t = r.signal.table
        assert t.n == ev.n_total
        assert t.a + t.b == ev.drug_totals["alpha"]
        assert t.a + t.c == ev.reaction_totals[r.reaction]


def test_stoplisted_terms_are_dropped_before_the_top_n_cut():
    ev = make_events()  # stoplisted term is first in the source's list
    one = analyze_drug(ev, "alpha", top_n=1)
    assert len(one) == 1
    assert one[0].reaction != "DRUG INEFFECTIVE"
    assert "DRUG INEFFECTIVE" not in [r.reaction for r in analyze_drug(ev, "alpha")]


def test_stoplist_matching_ignores_case():
    ev = make_events()
    ev.top["alpha"][0] = "Drug Ineffective"
    assert all(r.reaction.upper() != "DRUG INEFFECTIVE" for r in analyze_drug(ev, "alpha"))


def test_results_sorted_by_prr_descending_with_undefined_last():
    results = analyze_drug(make_events(), "alpha")
    prrs = [r.signal.prr for r in results]
    defined = [p for p in prrs if p is not None]
    assert defined == sorted(defined, reverse=True)
    assert prrs[len(defined):] == [None] * (len(prrs) - len(defined))
    assert any(p is None for p in prrs)


def test_flag_follows_the_screening_rule():
    for r in analyze_drug(make_events(), "alpha"):
        s = r.signal
        expected = s.prr is not None and s.prr >= 2 and s.chi2 >= 4 and s.table.a >= 3
        assert s.flagged == expected


def test_clear_signal_is_flagged_and_null_reaction_is_not():
    by_name = {r.reaction: r.signal for r in analyze_drug(make_events(), "alpha")}
    assert by_name["RASH"].flagged is True
    assert by_name["NAUSEA"].flagged is False


def test_unknown_drug_gives_empty_result():
    assert analyze_drug(make_events(), "nonexistent") == []
