import pytest

from safety_signal.checks import check_text
from safety_signal.facts import Facts, FindingFact
from tests.fakes import GOOD_SUMMARY, make_facts

CITED = ["LACTIC ACIDOSIS", "ACUTE KIDNEY INJURY"]


def test_grounded_text_passes():
    assert check_text(GOOD_SUMMARY, CITED, make_facts()) == []


def test_invented_number_is_caught():
    text = GOOD_SUMMARY.replace("19,398", "20,000")
    assert any("20,000" in p for p in check_text(text, CITED, make_facts()))


def test_percentages_are_never_allowed():
    assert any("%" in p for p in check_text(GOOD_SUMMARY + " About 5% of reports.", CITED, make_facts()))


def test_trailing_zero_formats_are_accepted():
    assert check_text(GOOD_SUMMARY.replace("72.89", "72.890"), CITED, make_facts()) == []


def test_date_numbers_come_from_as_of():
    text = GOOD_SUMMARY + " Data as of July 30, 2026."
    assert check_text(text, CITED, make_facts()) == []
    assert check_text(GOOD_SUMMARY + " Data as of July 31, 2026.", CITED, make_facts())


def test_digits_inside_reaction_names_are_not_numeric_claims():
    facts = Facts("alpha", None, 1, (FindingFact("TYPE 2 DIABETES MELLITUS", 5, 2.5, 9.0, True, "candidate"),))
    text = "Of 1 reactions analysed, 1 was flagged: TYPE 2 DIABETES MELLITUS with 5 reports and PRR 2.50 was not found in the sampled label text."
    assert check_text(text, ["TYPE 2 DIABETES MELLITUS"], facts) == []


@pytest.mark.parametrize(
    "phrase",
    [
        "Alpha causes lactic acidosis.",
        "This proves the link.",
        "Results were confirmed.",
        "Patients should stop taking it.",
        "A new risk was found.",
        "The incidence is high.",
        "The drug is safe.",
        "This is the most dangerous reaction.",
        "Ask your doctor, we recommend care.",
        "A lower dose helps.",
    ],
)
def test_forbidden_wording_is_caught(phrase):
    assert any("forbidden" in p for p in check_text(GOOD_SUMMARY + " " + phrase, CITED, make_facts()))


def test_words_inside_reaction_names_are_not_flagged():
    facts = Facts("alpha", None, 1, (FindingFact("DRUG DOSE OMISSION", 5, 2.5, 9.0, True, "candidate"),))
    text = "Of 1 reactions analysed, 1 was flagged: DRUG DOSE OMISSION with 5 reports and PRR 2.50 was not found in the sampled label text."
    assert check_text(text, ["DRUG DOSE OMISSION"], facts) == []


def test_every_flagged_reaction_must_be_mentioned():
    text = "Of 3 reactions analysed, 2 were flagged. LACTIC ACIDOSIS (19,398 reports) was found in the sampled label text."
    assert any("ACUTE KIDNEY INJURY" in p for p in check_text(text, CITED, make_facts()))
    assert check_text(text, CITED, make_facts(), require_flagged=False) == []


def test_cited_reaction_must_exist_in_facts():
    assert any("HEADACHE" in p for p in check_text(GOOD_SUMMARY, CITED + ["HEADACHE"], make_facts()))


def test_length_limit():
    assert any("words" in p for p in check_text(GOOD_SUMMARY + " word" * 160, CITED, make_facts()))


def test_spelling_variants_of_a_reaction_name_are_accepted():
    facts = Facts("alpha", None, 1, (FindingFact("DIARRHOEA", 28201, 2.14, 16977.3, True, "labeled"),))
    text = "Of 1 reactions analysed, 1 was flagged. Diarrhea (28,201 reports, PRR 2.14) was found in the sampled label text."
    assert check_text(text, ["Diarrhea"], facts) == []
    assert check_text(text.replace("Diarrhea", "Nausea"), ["Nausea"], facts)


from safety_signal.summary import template_body  # noqa: E402


def test_live_failure_wrong_label_status_for_a_labeled_reaction_is_caught():
    # Real model output pattern from the first working run: a labeled reaction listed as "not found".
    text = ("Of 3 reactions analysed, 2 were flagged. LACTIC ACIDOSIS (19398 reports, PRR 72.89) and "
            "ACUTE KIDNEY INJURY (18251 reports, PRR 6.37) were not found in the sampled label text.")
    problems = check_text(text, CITED, make_facts())
    assert any("LACTIC ACIDOSIS" in p and "label_status" in p for p in problems)
    assert not any("ACUTE KIDNEY INJURY" in p and "is described" in p for p in problems)


def test_one_sentence_with_two_statuses_is_rejected():
    text = ("Of 3 reactions analysed, 2 were flagged. LACTIC ACIDOSIS was found in the sampled label text, "
            "while ACUTE KIDNEY INJURY was not found in the sampled label text.")
    assert any("different label statuses" in p for p in check_text(text, CITED, make_facts()))


def test_flagged_reaction_without_a_status_statement_is_rejected():
    text = "Of 3 reactions analysed, 2 were flagged. LACTIC ACIDOSIS and ACUTE KIDNEY INJURY were flagged."
    problems = check_text(text, CITED, make_facts())
    assert sum("does not state the correct label status" in p for p in problems) == 2


def test_contained_reaction_names_are_not_double_counted():
    facts = Facts("alpha", None, 1, (
        FindingFact("PAIN IN EXTREMITY", 9, 2.5, 9.0, True, "labeled"),
        FindingFact("PAIN", 8, 2.5, 9.0, True, "candidate"),
    ))
    text = ("Of 2 reactions analysed, 2 were flagged. PAIN IN EXTREMITY (9 reports, PRR 2.50) was found in the sampled label text. "
            "PAIN (8 reports, PRR 2.50) was not found in the sampled label text.")
    assert check_text(text, ["PAIN IN EXTREMITY", "PAIN"], facts) == []


def test_numbers_glued_to_letters_are_still_checked():
    assert any("9.99" in p for p in check_text(GOOD_SUMMARY.replace("PRR 72.89", "PRR9.99"), CITED, make_facts()))


def test_the_template_fallback_satisfies_every_check():
    facts = make_facts()
    assert check_text(template_body(facts), [], facts) == []


def test_each_flagged_reaction_must_come_with_its_reports_and_prr():
    text = ("Of 3 reactions analysed, 2 were flagged. LACTIC ACIDOSIS was found in the sampled label text. "
            "ACUTE KIDNEY INJURY (18,251 reports, PRR 6.37) was not found in the sampled label text.")
    problems = check_text(text, CITED, make_facts())
    assert any("LACTIC ACIDOSIS" in p and "reports and PRR" in p for p in problems)
    assert not any("ACUTE KIDNEY INJURY" in p and "reports and PRR" in p for p in problems)
    # answers to questions are not required to repeat the numbers
    assert check_text("LACTIC ACIDOSIS was found in the sampled label text.", ["LACTIC ACIDOSIS"], make_facts(), require_flagged=False) == []


def test_a_reaction_named_alongside_a_longer_one_in_the_same_sentence_still_counts():
    facts = Facts("alpha", None, 1, (
        FindingFact("PAIN IN EXTREMITY", 9, 2.5, 9.0, True, "candidate"),
        FindingFact("PAIN", 8, 2.5, 9.0, True, "candidate"),
    ))
    text = ("Of 2 reactions analysed, 2 were flagged. PAIN IN EXTREMITY (9 reports, PRR 2.50) and PAIN (8 reports, "
            "PRR 2.50) were not found in the sampled label text.")
    assert check_text(text, ["PAIN IN EXTREMITY", "PAIN"], facts) == []


def test_the_shorter_reaction_is_still_required_when_only_the_longer_name_is_given():
    facts = Facts("alpha", None, 1, (
        FindingFact("PAIN IN EXTREMITY", 9, 2.5, 9.0, True, "candidate"),
        FindingFact("PAIN", 8, 2.5, 9.0, True, "candidate"),
    ))
    text = "Of 2 reactions analysed, 2 were flagged. PAIN IN EXTREMITY (9 reports, PRR 2.50) was not found in the sampled label text."
    assert any("'PAIN'" in p for p in check_text(text, ["PAIN IN EXTREMITY"], facts))
