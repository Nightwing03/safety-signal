"""Decides whether a reaction term already appears in a drug's label text.

Result per reaction: "labeled" (found in at least min_share of the single-ingredient label records),
"candidate" (not found), or "no_label_data" (no usable label records)."""
from dataclasses import dataclass

from safety_signal.labels import Label
from safety_signal.text import contains_phrase, contains_standalone, normalize

# MedDRA preferred term (normalised) -> other phrasings used in label prose. Hand-written; the
# evaluation measures how good it is. Do not add a phrase just because it makes one case pass.
ALIASES = {
    "acute kidney injury": ("acute renal failure", "renal failure", "kidney injury", "acute kidney failure"),
    "weight decreased": ("weight loss", "decreased weight"),
    "decreased appetite": ("anorexia", "loss of appetite", "appetite loss"),
    "blood glucose increased": ("hyperglycemia", "increased blood glucose", "elevated blood glucose"),
    "dizziness": ("dizzy", "lightheaded", "light headed"),  # lightheaded added: dev miss, metformin
    "fatigue": ("tiredness",),
    "asthenia": (),  # 'weakness' removed: it matched 'muscle weakness' (dev miss: atorvastatin)
    "vomiting": ("emesis",),
}


# Single-word terms so generic that they appear inside many specific terms ('abdominal pain', 'back pain').
# Added from dev evidence (metformin, atorvastatin); only a standalone mention counts.
GENERIC_TERMS = frozenset({"pain"})


@dataclass(frozen=True)
class LabelMatch:
    reaction: str
    records_checked: int
    records_matched: int
    sections: tuple  # section names where it was found in at least one record

    @property
    def share(self) -> float:
        return self.records_matched / self.records_checked if self.records_checked else 0.0

    def status(self, min_share: float = 0.5) -> str:
        if self.records_checked == 0:
            return "no_label_data"
        return "labeled" if self.share >= min_share else "candidate"


class LabelIndex:
    """Normalises label text once so many reactions can be matched cheaply."""

    def __init__(self, labels: list[Label], aliases: dict = ALIASES):
        self._docs = [{name: normalize(text) for name, text in lab.sections.items()} for lab in labels]
        self._aliases = {normalize(k): v for k, v in aliases.items()}

    def __len__(self) -> int:
        return len(self._docs)

    def match(self, reaction: str) -> LabelMatch:
        key = normalize(reaction)
        phrases = (reaction, *self._aliases.get(key, ()))
        check = contains_standalone if key in GENERIC_TERMS else contains_phrase
        matched = 0
        sections: set = set()
        for doc in self._docs:
            hit = {name for name, text in doc.items() if any(check(text, p) for p in phrases)}
            if hit:
                matched += 1
                sections |= hit
        return LabelMatch(reaction, len(self._docs), matched, tuple(sorted(sections)))
