import csv
import io

from safety_signal.evaluation import (
    Metrics,
    main,
    pick_reference,
    score_pairs,
    snippets,
    stems,
)
from tests.fakes import FakeLabels, make_events, make_label


def test_stems_use_longest_content_words_and_skip_filler():
    assert stems("BLOOD GLUCOSE INCREASED") == ["gluco"]
    assert stems("ACUTE KIDNEY INJURY") == ["kidne", "injur"]
    assert "diarr" in stems("DIARRHOEA")


def test_snippets_show_context_and_cap_the_count():
    text = "x " * 100 + "severe diarrhea occurred " + "y " * 100 + "diarrhea again " + "z " * 100 + "diarrhea third"
    out = snippets({"adverse_reactions": text}, "DIARRHOEA", max_n=2)
    assert len(out) == 2 and all("diarrhea" in s for s in out)


def test_snippets_empty_when_term_absent():
    assert snippets({"adverse_reactions": "only headache"}, "RASH") == []


def test_pick_reference_is_deterministic():
    labels = [make_label(set_id="b"), make_label(set_id="a"), make_label(set_id="c")]
    assert pick_reference(labels).set_id == "a"
    assert pick_reference(list(reversed(labels))).set_id == "a"


def test_metrics_arithmetic():
    m = score_pairs([(True, True), (True, True), (True, False), (False, True), (False, False)])
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 1)
    assert m.precision == 2 / 3 and m.recall == 2 / 3 and m.n == 5


def test_metrics_undefined_ratios_are_none():
    m = Metrics()
    assert m.precision is None and m.recall is None


def run(argv, labels):
    buf = io.StringIO()
    code = main(argv, source_factory=lambda a: make_events(), label_factory=lambda a: FakeLabels(labels), out=buf)
    return code, buf.getvalue()


LABELS = [make_label(set_id="r1", adverse_reactions="Reported: rash and nausea.")]


def test_prepare_writes_blank_sheet_without_leaking_matcher_answer(tmp_path):
    out = tmp_path / "eval"
    code, _ = run(["prepare", "alpha", "--out", str(out)], LABELS)
    assert code == 0
    text = (out / "alpha.csv").read_text()
    assert "labeled" not in text and "candidate" not in text
    rows = list(csv.DictReader(open(out / "alpha.csv")))
    assert {r["reaction"] for r in rows} == {"NAUSEA", "RASH", "RARE"}
    assert all(r["human"] == "" for r in rows)
    assert (out / "alpha_label.txt").exists()


def fill(path, answers):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["human"] = answers.get(r["reaction"], "")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_score_perfect_agreement_and_skips_blank_rows(tmp_path):
    out = tmp_path / "eval"
    run(["prepare", "alpha", "--out", str(out)], LABELS)
    fill(out / "alpha.csv", {"RASH": "y", "NAUSEA": "Y", "RARE": "n"})
    _, text = run(["score", str(out / "alpha.csv")], LABELS)
    assert "precision=1.00 recall=1.00" in text
    fill(out / "alpha.csv", {"RASH": "y"})
    _, text = run(["score", str(out / "alpha.csv")], LABELS)
    assert "2 rows had no y/n answer" in text


def test_score_reports_disagreements_and_splits_holdout(tmp_path):
    out = tmp_path / "eval"
    run(["prepare", "alpha", "--out", str(out)], LABELS)
    fill(out / "alpha.csv", {"RASH": "n", "NAUSEA": "y", "RARE": "y"})
    _, text = run(["score", str(out / "alpha.csv"), "--holdout", "alpha"], LABELS)
    assert text.splitlines()[0].startswith("dev") and "n=0" in text.splitlines()[0]
    assert "holdout" in text and "human=no matcher=yes" in text and "human=yes matcher=no" in text
