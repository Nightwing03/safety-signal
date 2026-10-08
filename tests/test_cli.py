import io

from safety_signal.cli import main
from safety_signal.errors import DataError
from tests.fakes import FakeLabels, make_events, make_label


def run(argv, factory, labels=None):
    buf = io.StringIO()
    code = main(argv, source_factory=factory, label_factory=lambda a: FakeLabels(labels or []), out=buf)
    return code, buf.getvalue()


def test_label_column_distinguishes_labeled_candidate_and_no_data():
    labels = [make_label(adverse_reactions="Reported: nausea and rash.")]
    _, text = run(["alpha"], lambda a: make_events(), labels)
    row = {line.split()[0]: line for line in text.splitlines() if line[:1].isupper() and "SIGNAL" in line or line.startswith("NAUSEA")}
    assert row["RASH"].rstrip().endswith("labeled")
    assert row["NAUSEA"].rstrip().endswith("labeled")
    _, text = run(["alpha"], lambda a: make_events(), [make_label(adverse_reactions="Only headache.")])
    assert any(line.startswith("RASH") and line.rstrip().endswith("candidate") for line in text.splitlines())
    _, text = run(["alpha"], lambda a: make_events(), [])
    assert any(line.startswith("RASH") and line.rstrip().endswith("no_label_data") for line in text.splitlines())


def test_no_labels_flag_skips_the_label_source():
    def boom(args):
        raise AssertionError("label source must not be built")

    buf = io.StringIO()
    code = main(["alpha", "--no-labels"], source_factory=lambda a: make_events(), label_factory=boom, out=buf)
    assert code == 0 and "RASH" in buf.getvalue()


def test_output_has_drug_as_of_date_caveat_and_flag():
    code, text = run(["alpha", "--top", "5"], lambda args: make_events())
    assert code == 0
    assert "alpha" in text
    assert "data as of: 2026-01-01" in text
    assert "voluntary" in text
    assert "RASH" in text and "SIGNAL" in text
    assert "DRUG INEFFECTIVE" not in text


def test_unknown_drug_prints_a_message_not_a_crash():
    code, text = run(["nonexistent"], lambda args: make_events())
    assert code == 0
    assert "No reactions found" in text


def test_library_errors_become_exit_code_one(capsys):
    def factory(args):
        raise DataError("boom")

    code, _ = run(["alpha"], factory)
    assert code == 1
    assert "error: boom" in capsys.readouterr().err


def test_summary_flag_prints_a_checked_summary_and_default_does_not_call_the_model():
    from safety_signal.summary import SummaryOut
    from tests.fakes import FakeLLM

    text = "Of 3 reactions analysed, 1 was flagged. RASH (100 reports, PRR 5.21) was found in the sampled label text."
    llm = FakeLLM([SummaryOut(summary=text, cited_reactions=["RASH"])])
    labels = [make_label(adverse_reactions="rash")]
    buf = io.StringIO()
    code = main(["alpha", "--summary"], source_factory=lambda a: make_events(),
                label_factory=lambda a: FakeLabels(labels), llm_factory=lambda a: llm, out=buf)
    assert code == 0 and "summary source: llm" in buf.getvalue() and text in buf.getvalue()

    def boom(args):
        raise AssertionError("model must not be built without --summary")

    buf = io.StringIO()
    assert main(["alpha"], source_factory=lambda a: make_events(),
                label_factory=lambda a: FakeLabels(labels), llm_factory=boom, out=buf) == 0


def test_failed_summary_attempts_are_shown_with_their_reasons():
    from safety_signal.summary import SummaryOut
    from tests.fakes import FakeLLM

    bad = SummaryOut(summary="RASH causes 999 problems.", cited_reactions=["RASH"])
    buf = io.StringIO()
    main(["alpha", "--summary"], source_factory=lambda a: make_events(),
         label_factory=lambda a: FakeLabels([make_label(adverse_reactions="rash")]),
         llm_factory=lambda a: FakeLLM([bad, bad]), out=buf)
    text = buf.getvalue()
    assert "summary source: template, attempts: 2" in text
    assert "attempt 1 failed checks:" in text and "attempt 2 failed checks:" in text and "999" in text
