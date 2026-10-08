import json

from safety_signal.facts import build_facts
from safety_signal.matcher import LabelIndex
from safety_signal.pairs import analyze_drug
from tests.fakes import make_events, make_label


def test_statuses_counts_and_label_records():
    results = analyze_drug(make_events(), "alpha")
    facts = build_facts("alpha", results, "2026-01-01", LabelIndex([make_label(adverse_reactions="rash")]))
    by = {f.reaction: f for f in facts.findings}
    assert by["RASH"].label_status == "labeled" and by["RASH"].flagged
    assert by["NAUSEA"].label_status == "candidate" and not by["NAUSEA"].flagged
    assert facts.label_records == 1
    c = facts.counts()
    assert c["reactions_analysed"] == 3 and c["flagged"] == 1
    assert c["flagged_found_in_label"] == 1 and c["flagged_not_found_in_label"] == 0


def test_without_labels_everything_is_no_label_data():
    facts = build_facts("alpha", analyze_drug(make_events(), "alpha"), None, None)
    assert {f.label_status for f in facts.findings} == {"no_label_data"}
    assert facts.label_records == 0 and facts.as_of is None


def test_json_carries_rounded_values_and_all_reactions():
    facts = build_facts("alpha", analyze_drug(make_events(), "alpha"), "2026-01-01", None)
    data = json.loads(facts.to_json())
    rash = next(r for r in data["reactions"] if r["reaction"] == "RASH")
    assert rash["prr"] == 5.21 and rash["reports"] == 100
    assert len(data["reactions"]) == 3 and data["data_as_of"] == "2026-01-01"


def test_flagged_reactions_are_pre_grouped_by_label_status_for_the_model():
    from tests.fakes import make_facts

    data = json.loads(make_facts().to_json())
    assert data["flagged_by_label_status"] == {
        "labeled": ["LACTIC ACIDOSIS"], "candidate": ["ACUTE KIDNEY INJURY"], "no_label_data": [],
    }
