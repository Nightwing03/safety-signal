import pytest

from safety_signal.text import contains_phrase, normalize


def test_british_spellings_map_to_us():
    assert normalize("DIARRHOEA") == "diarrhea"
    assert normalize("Dyspnoea, oedema") == "dyspnea edema"


def test_punctuation_and_case_collapse():
    assert normalize("  Lactic-Acidosis;  (5.1) ") == "lactic acidosis 5 1"


def test_phrase_matches_whole_words_only():
    text = normalize("Patients reported painful swelling and abdominal pain.")
    assert contains_phrase(text, "pain")
    assert not contains_phrase(normalize("painful swelling"), "pain")
    assert not contains_phrase(normalize("Bipolar disorder"), "polar disorders")


@pytest.mark.parametrize("phrase", ["", "   ", "---"])
def test_empty_phrase_never_matches(phrase):
    assert contains_phrase(normalize("anything at all"), phrase) is False


from safety_signal.text import contains_standalone  # noqa: E402


def test_standalone_rules():
    n = normalize
    assert contains_standalone(n("Pain 1 2"), "pain")
    assert contains_standalone(n("fatigue 5 pain 6"), "pain")
    assert not contains_standalone(n("abdominal pain"), "pain")
    assert not contains_standalone(n("pain in extremity"), "pain")
    assert not contains_standalone(n("painful"), "pain")
    assert contains_standalone(n("back pain and then 7 pain 8"), "pain")
    assert not contains_standalone(n("anything"), "")
