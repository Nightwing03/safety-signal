"""Drug label source (openFDA drug/label.json). Pulls only the sections that list safety information."""
from dataclasses import dataclass, field
from typing import Optional, Protocol

from safety_signal.errors import NotFoundError
from safety_signal.events import quote_term

LABEL_ENDPOINT = "drug/label.json"
LABEL_FIELD = "openfda.generic_name"
# `warnings` is the older-format equivalent of warnings_and_cautions.
SECTIONS = ("boxed_warning", "warnings_and_cautions", "warnings", "adverse_reactions", "contraindications")


@dataclass(frozen=True)
class Label:
    set_id: str
    generic_names: tuple
    brand_names: tuple
    effective_time: Optional[str]
    sections: dict = field(default_factory=dict)


class LabelSource(Protocol):
    def labels(self, drug: str) -> list[Label]: ...


def parse_label(record: dict) -> Label:
    openfda = record.get("openfda") or {}
    sections = {k: " ".join(record[k]) for k in SECTIONS if isinstance(record.get(k), list)}
    return Label(
        set_id=record.get("set_id", ""),
        generic_names=tuple(openfda.get("generic_name") or ()),
        brand_names=tuple(openfda.get("brand_name") or ()),
        effective_time=record.get("effective_time"),
        sections=sections,
    )


def is_single_ingredient(label: Label, drug: str) -> bool:
    """Combination products (e.g. 'SITAGLIPTIN AND METFORMIN') list other drugs' reactions; exclude them."""
    if not label.generic_names:
        return False
    d = drug.lower().strip()
    for name in label.generic_names:
        n = name.lower()
        if d not in n or " and " in n or "," in n or "/" in n:
            return False
    return True


class OpenFDALabels:
    """Fetches up to max_records label records for a drug. Records are large (about 170 KB each), so keep the cap low."""

    def __init__(self, client, page_size: int = 20, max_records: int = 40):
        self._client = client
        self._page = page_size
        self._max = max_records

    def labels(self, drug: str) -> list[Label]:
        search = f"{LABEL_FIELD}:{quote_term(drug)}"
        records: list[dict] = []
        skip = 0
        while len(records) < self._max:
            params = {"search": search, "limit": self._page, "skip": skip}
            try:
                body = self._client.get(LABEL_ENDPOINT, params)
            except NotFoundError:
                break
            page = body.get("results") or []
            if not page:
                break
            records.extend(page)
            skip += len(page)
            if len(page) < self._page:
                break
        parsed = (parse_label(r) for r in records[: self._max])
        return [lab for lab in parsed if is_single_ingredient(lab, drug)]
