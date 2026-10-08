"""One analysis run: counts -> signals -> label comparison -> facts. Shared by the API."""
from typing import Optional

from safety_signal.facts import Facts, build_facts
from safety_signal.matcher import LabelIndex
from safety_signal.pairs import analyze_drug


def analyse(source, label_source, drug: str, top_n: int) -> Optional[Facts]:
    """Returns None when the drug has no reports at all."""
    results = analyze_drug(source, drug, top_n=top_n)
    if not results:
        return None
    index = LabelIndex(label_source.labels(drug)) if label_source is not None else None
    return build_facts(drug, results, source.as_of(), index)
