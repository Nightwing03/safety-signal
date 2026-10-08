"""Text normalisation so MedDRA terms (often British spelling) can be compared with label prose (US spelling)."""
import re

# Substring replacements, applied after lower-casing. Extend when the evaluation shows a miss.
BRITISH_TO_US = {
    "diarrhoea": "diarrhea",
    "dyspnoea": "dyspnea",
    "oedema": "edema",
    "haemorrhage": "hemorrhage",
    "haemolytic": "hemolytic",
    "haemolysis": "hemolysis",
    "anaemia": "anemia",
    "leucopenia": "leukopenia",
    "thrombocytopaenia": "thrombocytopenia",
    "paraesthesia": "paresthesia",
    "hypoaesthesia": "hypoesthesia",
    "oesophag": "esophag",
    "foetal": "fetal",
    "tumour": "tumor",
    "faeces": "feces",
    "dyspepsia": "dyspepsia",
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lower-case, map British spellings to US, reduce punctuation to single spaces."""
    t = text.lower()
    for british, us in BRITISH_TO_US.items():
        t = t.replace(british, us)
    return _NON_ALNUM.sub(" ", t).strip()


def contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Whole-word phrase match on already-normalised text ('pain' does not match 'painful')."""
    p = normalize(phrase)
    if not p:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", normalized_text) is not None


_NON_QUALIFIERS_AFTER = {"in", "of"}


def contains_standalone(normalized_text: str, phrase: str) -> bool:
    """Like contains_phrase, but only counts the phrase when it is not part of a more specific term.

    Rejected: preceded by a word ('abdominal pain', 'back pain') or followed by 'in'/'of' ('pain in extremity').
    Accepted: at the start of the text, or preceded by a number (a table row such as '5 pain 1 2')."""
    p = normalize(phrase)
    if not p:
        return False
    for m in re.finditer(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", normalized_text):
        before = normalized_text[: m.start()].split()
        after = normalized_text[m.end() :].split()
        if (not before or before[-1].isdigit()) and (not after or after[0] not in _NON_QUALIFIERS_AFTER):
            return True
    return False
