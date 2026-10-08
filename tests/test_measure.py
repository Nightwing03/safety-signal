import io
import json

from safety_signal.measure import aggregate, main, run_repeats
from safety_signal.summary import SummaryOut
from tests.fakes import GOOD_SUMMARY, FakeLLM, make_facts

CITED = ["LACTIC ACIDOSIS", "ACUTE KIDNEY INJURY"]
GOOD = SummaryOut(summary=GOOD_SUMMARY, cited_reactions=CITED)
BAD = SummaryOut(summary=GOOD_SUMMARY.replace("19,398", "20,000"), cited_reactions=CITED)


def test_rows_distinguish_first_attempt_retry_and_fallback():
    llm = FakeLLM([GOOD, BAD, GOOD, BAD, BAD])
    rows = run_repeats(make_facts(), llm, 3)
    a = aggregate(rows)
    assert (a["runs"], a["first_attempt_pass"], a["passed_after_retry"], a["template_fallback"]) == (3, 1, 1, 1)
    assert a["top_failure_reasons"]


def test_aggregate_of_nothing_does_not_crash():
    assert aggregate([]) == {"runs": 0}


def test_main_writes_one_json_line_per_run(tmp_path):
    out_file = tmp_path / "runs.jsonl"
    buf = io.StringIO()
    code = main(["alpha", "--repeats", "2", "--out", str(out_file)],
                build=lambda drug: (make_facts(drug), FakeLLM([GOOD, GOOD])), out=buf)
    lines = out_file.read_text().strip().splitlines()
    assert code == 0 and len(lines) == 2 and json.loads(lines[0])["drug"] == "alpha"
    assert "OVERALL runs 2  first-attempt 2" in buf.getvalue()


def test_drug_with_no_reports_is_skipped(tmp_path):
    buf = io.StringIO()
    main(["ghost", "--out", str(tmp_path / "r.jsonl")], build=lambda d: (None, None), out=buf)
    assert "skipped" in buf.getvalue() and not (tmp_path / "r.jsonl").exists()
