"""Turns counts from an EventSource into ranked drug-reaction signals."""
from dataclasses import dataclass
from typing import Iterable

from safety_signal.errors import DataError
from safety_signal.events import EventSource
from safety_signal.stats import Signal, Table, Thresholds, evaluate

# Terms that describe the report, not a clinical event. Matched case-insensitively.
DEFAULT_STOPLIST = frozenset(
    {
        "DRUG INEFFECTIVE",
        "OFF LABEL USE",
        "PRODUCT USE IN UNAPPROVED INDICATION",
        "PRODUCT DOSE OMISSION ISSUE",
        "INTENTIONAL PRODUCT MISUSE",
        "PRODUCT USE ISSUE",
        "WRONG TECHNIQUE IN PRODUCT USAGE PROCESS",
        "INAPPROPRIATE SCHEDULE OF PRODUCT ADMINISTRATION",
        "DRUG DOSE OMISSION",
        "PRODUCT QUALITY ISSUE",
    }
)


@dataclass(frozen=True)
class PairResult:
    drug: str
    reaction: str
    signal: Signal


def build_table(n_total: int, drug_total: int, reaction_total: int, pair: int) -> Table:
    """Derive all four cells from four counts. Raises DataError if they cannot be a valid table."""
    a = pair
    b = drug_total - a
    c = reaction_total - a
    d = n_total - a - b - c
    if min(a, b, c, d) < 0:
        raise DataError(
            f"inconsistent counts (total={n_total}, drug={drug_total}, reaction={reaction_total}, pair={pair})"
        )
    return Table(a, b, c, d)


def analyze_drug(
    source: EventSource,
    drug: str,
    top_n: int = 15,
    stoplist: Iterable[str] = DEFAULT_STOPLIST,
    thresholds: Thresholds = Thresholds(),
) -> list[PairResult]:
    blocked = {s.upper() for s in stoplist}
    reactions = [r for r in source.top_reactions(drug) if r.upper() not in blocked][:top_n]
    if not reactions:
        return []
    n_total = source.total()
    drug_total = source.drug_total(drug)
    results = []
    for reaction in reactions:
        table = build_table(
            n_total, drug_total, source.reaction_total(reaction), source.pair_count(drug, reaction)
        )
        results.append(PairResult(drug, reaction, evaluate(table, thresholds)))
    # Highest PRR first; undefined PRR last.
    results.sort(key=lambda r: (r.signal.prr is None, -(r.signal.prr or 0.0)))
    return results
