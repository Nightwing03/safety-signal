from safety_signal.matcher import LabelIndex
from tests.fakes import make_label


def idx(*texts, section="adverse_reactions"):
    return LabelIndex([make_label(set_id=str(i), **{section: t}) for i, t in enumerate(texts)])


def test_literal_term_is_labeled_and_section_is_reported():
    m = idx("Common reactions: nausea, vomiting.").match("NAUSEA")
    assert m.status() == "labeled" and m.sections == ("adverse_reactions",)


def test_british_term_matches_us_label_text():
    assert idx("diarrhea occurred").match("DIARRHOEA").status() == "labeled"


def test_alias_matches_different_wording():
    assert idx("Acute renal failure has been reported.").match("ACUTE KIDNEY INJURY").status() == "labeled"


def test_absent_term_is_candidate():
    assert idx("Only headache is listed.").match("RASH").status() == "candidate"


def test_partial_word_does_not_match():
    assert idx("The patient felt painful pressure.").match("PAIN").status() == "candidate"


def test_share_threshold_across_records():
    index = idx("rash reported", "nothing here", "nothing again")
    m = index.match("RASH")
    assert (m.records_checked, m.records_matched) == (3, 1)
    assert m.status(min_share=0.5) == "candidate"
    assert m.status(min_share=0.3) == "labeled"


def test_sections_are_collected_across_records():
    index = LabelIndex(
        [make_label(set_id="a", boxed_warning="lactic acidosis"), make_label(set_id="b", adverse_reactions="lactic acidosis")]
    )
    assert index.match("LACTIC ACIDOSIS").sections == ("adverse_reactions", "boxed_warning")


def test_no_records_means_no_label_data_not_candidate():
    assert LabelIndex([]).match("RASH").status() == "no_label_data"


def test_generic_term_ignores_specific_compounds():
    for text in ("abdominal pain was reported", "back pain 2 1", "Pain in extremity 6.0", "chest pain (by 2.1%)"):
        assert idx(text).match("PAIN").status() == "candidate", text


def test_generic_term_accepts_standalone_mentions():
    for text in ("Pain 1 2 3 4", "General Fatigue 11 8 Pain 6 3 Malaise", "pain"):
        assert idx(text).match("PAIN").status() == "labeled", text


def test_asthenia_does_not_match_muscle_weakness():
    assert idx("muscle weakness and tenderness").match("ASTHENIA").status() == "candidate"
    assert idx("asthenia 9 6").match("ASTHENIA").status() == "labeled"


def test_lightheaded_counts_for_dizziness():
    assert idx("adverse reactions: lightheaded").match("DIZZINESS").status() == "labeled"
