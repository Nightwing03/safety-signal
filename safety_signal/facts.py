"""The only thing the language model ever sees: computed facts, nothing else."""
import json
from dataclasses import dataclass
from typing import Optional

from safety_signal.matcher import LabelIndex
from safety_signal.pairs import PairResult


@dataclass(frozen=True)
class FindingFact:
    reaction: str
    reports: int
    prr: Optional[float]
    chi2: Optional[float]
    flagged: bool
    label_status: str  # "labeled" | "candidate" | "no_label_data"
    label_sections: tuple = ()


@dataclass(frozen=True)
class Facts:
    drug: str
    as_of: Optional[str]
    label_records: int
    findings: tuple

    @property
    def flagged(self) -> tuple:
        return tuple(f for f in self.findings if f.flagged)

    def counts(self) -> dict:
        fl = self.flagged
        return {
            "reactions_analysed": len(self.findings),
            "flagged": len(fl),
            "flagged_found_in_label": sum(1 for f in fl if f.label_status == "labeled"),
            "flagged_not_found_in_label": sum(1 for f in fl if f.label_status == "candidate"),
            "flagged_label_not_checked": sum(1 for f in fl if f.label_status == "no_label_data"),
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "drug": self.drug,
                "data_as_of": self.as_of,
                "label_records_checked": self.label_records,
                "counts": self.counts(),
                "flagged_by_label_status": {
                    status: [f.reaction for f in self.flagged if f.label_status == status]
                    for status in ("labeled", "candidate", "no_label_data")
                },
                "reactions": [
                    {
                        "reaction": f.reaction,
                        "reports": f.reports,
                        "prr": None if f.prr is None else round(f.prr, 2),
                        "chi2": None if f.chi2 is None else round(f.chi2, 1),
                        "flagged": f.flagged,
                        "label_status": f.label_status,
                        "label_sections": list(f.label_sections),
                    }
                    for f in self.findings
                ],
            },
            indent=1,
        )


def build_facts(drug: str, results: list[PairResult], as_of: Optional[str], index: Optional[LabelIndex]) -> Facts:
    findings = []
    for r in results:
        if index is None:
            status, sections = "no_label_data", ()
        else:
            m = index.match(r.reaction)
            status, sections = m.status(), m.sections
        s = r.signal
        findings.append(FindingFact(r.reaction, s.table.a, s.prr, s.chi2, s.flagged, status, sections))
    return Facts(drug, as_of, len(index) if index is not None else 0, tuple(findings))
