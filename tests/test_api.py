from fastapi.testclient import TestClient

from safety_signal.api import create_app
from safety_signal.errors import DataError
from safety_signal.summary import REFUSAL, AnswerOut, SummaryOut
from tests.fakes import FakeLabels, FakeLLM, make_events, make_label

LABELS = [make_label(adverse_reactions="rash")]
GOOD = ("Of 3 reactions analysed, 1 was flagged. RASH (100 reports, PRR 5.21) was found in the sampled label text.")


def client(llm_items=(), labels=LABELS, source=None, llm_factory=None):
    llm = FakeLLM(llm_items)
    app = create_app(source or make_events(), FakeLabels(labels), llm_factory or (lambda: llm))
    return TestClient(app), llm


class Boom:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise DataError("secret internal detail")
        return fail


def test_health():
    c, _ = client()
    assert c.get("/health").json() == {"status": "ok"}


def test_signals_returns_facts_with_grouping_and_as_of():
    c, _ = client()
    body = c.get("/signals/alpha").json()
    assert body["data_as_of"] == "2026-01-01" and body["counts"]["flagged"] == 1
    assert body["flagged_by_label_status"]["labeled"] == ["RASH"]
    assert {r["reaction"] for r in body["reactions"]} == {"RASH", "NAUSEA", "RARE"}


def test_unknown_drug_is_404():
    c, _ = client()
    assert c.get("/signals/nonexistent").status_code == 404


def test_bad_input_is_rejected_before_any_upstream_call():
    c, _ = client(source=Boom())
    assert c.get('/signals/bad%22name').status_code == 422
    assert c.get("/signals/alpha?top=0").status_code == 422
    assert c.get("/signals/alpha?top=31").status_code == 422


def test_summary_uses_model_and_reports_metadata():
    c, llm = client([SummaryOut(summary=GOOD, cited_reactions=["RASH"])])
    body = c.get("/summary/alpha").json()
    assert body["source"] == "llm" and body["attempts"] == 1 and body["drug"] == "alpha"
    assert GOOD in body["text"] and body["input_tokens"] == 100 and len(llm.calls) == 1


def test_summary_falls_back_to_template_when_the_model_cannot_be_built():
    def broken():
        raise ImportError("no model client")

    c, _ = client(llm_factory=broken)
    r = c.get("/summary/alpha")
    assert r.status_code == 200 and r.json()["source"] == "template" and "RASH" in r.json()["text"]


def test_upstream_failure_is_502_without_leaking_details():
    c, _ = client(source=Boom())
    r = c.get("/signals/alpha")
    assert r.status_code == 502 and "secret" not in r.text


def test_out_of_scope_question_is_refused_without_touching_data_or_model():
    c, llm = client(source=Boom())
    r = c.post("/ask", json={"drug": "alpha", "question": "Should I stop taking alpha?"})
    assert r.status_code == 200 and r.json()["source"] == "refusal" and r.json()["text"] == REFUSAL
    assert llm.calls == []


def test_in_scope_question_gets_a_checked_answer():
    ok = AnswerOut(answer="RASH has 100 reports.", cited_reactions=["RASH"])
    c, _ = client([ok])
    body = c.post("/ask", json={"drug": "alpha", "question": "How many reports of RASH?"}).json()
    assert body["source"] == "llm" and "100" in body["text"]


def test_ask_validates_the_body():
    c, _ = client()
    assert c.post("/ask", json={"drug": "alpha", "question": ""}).status_code == 422
    assert c.post("/ask", json={"drug": "alpha"}).status_code == 422
