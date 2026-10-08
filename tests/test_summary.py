import json

import pytest

from safety_signal.summary import (
    CAVEAT, OUT_OF_SCOPE, REFUSAL, AnswerOut, SummaryOut, answer_question, out_of_scope, summarize,
)
from tests.fakes import GOOD_SUMMARY, FakeLLM, make_facts

CITED = ["LACTIC ACIDOSIS", "ACUTE KIDNEY INJURY"]
GOOD = SummaryOut(summary=GOOD_SUMMARY, cited_reactions=CITED)
BAD = SummaryOut(summary=GOOD_SUMMARY.replace("19,398", "20,000"), cited_reactions=CITED)


def test_valid_output_is_used_with_header_and_caveat():
    llm = FakeLLM([GOOD])
    r = summarize(make_facts(), llm)
    assert r.source == "llm" and r.attempts == 1 and r.problems == []
    assert "data as of 2026-07-30" in r.text and GOOD_SUMMARY in r.text and CAVEAT in r.text
    assert (r.provider, r.model, r.input_tokens, r.output_tokens) == ("fake", "fake-1", 100, 50)


def test_prompt_contains_only_the_facts_json():
    llm = FakeLLM([GOOD])
    summarize(make_facts(), llm)
    user = llm.calls[0]["user"]
    data = json.loads(user.split("\n", 1)[1])
    assert len(data["reactions"]) == 3 and data["counts"]["flagged"] == 2
    assert "failed these checks" not in user


def test_failed_check_triggers_one_retry_that_includes_the_problem():
    llm = FakeLLM([BAD, GOOD])
    r = summarize(make_facts(), llm)
    assert r.source == "llm" and r.attempts == 2 and len(r.problems) == 1
    assert "failed these checks" in llm.calls[1]["user"] and "20,000" in llm.calls[1]["user"]
    assert r.input_tokens == 200  # usage accumulates across attempts


def test_two_failures_fall_back_to_template_and_model_text_is_discarded():
    r = summarize(make_facts(), FakeLLM([BAD, BAD]))
    assert r.source == "template" and r.attempts == 2 and len(r.problems) == 2
    assert "20,000" not in r.text
    assert "LACTIC ACIDOSIS: 19398 reports, PRR 72.89" in r.text and "not found in the sampled label text" in r.text
    assert CAVEAT in r.text


def test_provider_failure_falls_back_to_template():
    r = summarize(make_facts(), FakeLLM([RuntimeError("down")]))
    assert r.source == "template" and r.error == "RuntimeError" and r.attempts == 1
    assert "LACTIC ACIDOSIS" in r.text


def test_nothing_flagged_never_calls_the_model():
    facts = make_facts()
    quiet = type(facts)(facts.drug, facts.as_of, facts.label_records, tuple(
        type(f)(f.reaction, f.reports, f.prr, f.chi2, False, f.label_status) for f in facts.findings))
    r = summarize(quiet, FakeLLM([]))
    assert r.source == "template" and r.attempts == 0 and "none met the screening rule" in r.text


@pytest.mark.parametrize(
    "q",
    [
        "Should I stop taking alpha?", "What dose is safe?", "Is it safe in pregnancy?",
        "Does alpha cause lactic acidosis?", "Can I take it with alcohol?", "Is alpha better than beta?",
        "Why does it happen?", "What are the symptoms?",
    ],
)
def test_out_of_scope_questions_are_refused_without_calling_the_model(q):
    assert out_of_scope(q)
    llm = FakeLLM([])
    r = answer_question(q, make_facts(), llm)
    assert r.source == "refusal" and r.text == REFUSAL and llm.calls == []


@pytest.mark.parametrize(
    "q",
    ["Which reactions were flagged?", "How many reports of LACTIC ACIDOSIS?", "Was ACUTE KIDNEY INJURY found in the label?"],
)
def test_in_scope_questions_are_not_pre_filtered(q):
    assert not out_of_scope(q)


def test_empty_question_is_refused():
    assert answer_question("   ", make_facts(), FakeLLM([])).source == "refusal"


def test_model_can_refuse_with_the_sentinel():
    r = answer_question("Tell me about alpha", make_facts(), FakeLLM([AnswerOut(answer=OUT_OF_SCOPE)]))
    assert r.source == "refusal" and r.text == REFUSAL


def test_grounded_answer_passes_and_ungrounded_answer_is_not_shown():
    ok = AnswerOut(answer="LACTIC ACIDOSIS has 19,398 reports.", cited_reactions=["LACTIC ACIDOSIS"])
    r = answer_question("How many reports of LACTIC ACIDOSIS?", make_facts(), FakeLLM([ok]))
    assert r.source == "llm" and "19,398" in r.text
    bad = AnswerOut(answer="LACTIC ACIDOSIS has 99,999 reports.", cited_reactions=["LACTIC ACIDOSIS"])
    r = answer_question("How many reports of LACTIC ACIDOSIS?", make_facts(), FakeLLM([bad, bad]))
    assert r.source == "refusal" and "99,999" not in r.text and len(r.problems) == 2
