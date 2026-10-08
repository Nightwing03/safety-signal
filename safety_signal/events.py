"""Adverse-event source. EventSource is the interface the rest of the package depends on;
OpenFDAEvents is the only place that knows openFDA field names and query syntax."""
from typing import Optional, Protocol

from safety_signal.errors import DataError, NotFoundError

EVENT_ENDPOINT = "drug/event.json"
DRUG_FIELD = "patient.drug.openfda.generic_name"
REACTION_FIELD = "patient.reaction.reactionmeddrapt.exact"


class EventSource(Protocol):
    def total(self) -> int: ...
    def drug_total(self, drug: str) -> int: ...
    def reaction_total(self, reaction: str) -> int: ...
    def pair_count(self, drug: str, reaction: str) -> int: ...
    def top_reactions(self, drug: str) -> list[str]: ...
    def as_of(self) -> Optional[str]: ...


def _quote(value: str) -> str:
    if '"' in value or not value.strip():
        raise ValueError(f"invalid search term: {value!r}")
    return f'"{value.strip()}"'


class OpenFDAEvents:
    def __init__(self, client):
        self._client = client
        self._as_of: Optional[str] = None

    def _body(self, params: dict) -> dict:
        body = self._client.get(EVENT_ENDPOINT, params)
        meta = body.get("meta") if isinstance(body, dict) else None
        if not meta:
            raise DataError("response has no meta block")
        self._as_of = meta.get("last_updated", self._as_of)
        return body

    def _total(self, search: Optional[str]) -> int:
        params: dict = {"limit": 1}
        if search:
            params["search"] = search
        try:
            body = self._body(params)
        except NotFoundError:
            return 0
        try:
            return int(body["meta"]["results"]["total"])
        except (KeyError, TypeError, ValueError):
            raise DataError("response has no meta.results.total") from None

    def total(self) -> int:
        return self._total(None)

    def drug_total(self, drug: str) -> int:
        return self._total(f"{DRUG_FIELD}:{_quote(drug)}")

    def reaction_total(self, reaction: str) -> int:
        return self._total(f"{REACTION_FIELD}:{_quote(reaction)}")

    def pair_count(self, drug: str, reaction: str) -> int:
        return self._total(f"{DRUG_FIELD}:{_quote(drug)} AND {REACTION_FIELD}:{_quote(reaction)}")

    def top_reactions(self, drug: str) -> list[str]:
        params = {"search": f"{DRUG_FIELD}:{_quote(drug)}", "count": REACTION_FIELD}
        try:
            body = self._body(params)
        except NotFoundError:
            return []
        try:
            return [row["term"] for row in body["results"]]
        except (KeyError, TypeError):
            raise DataError("count response has no results/term") from None

    def as_of(self) -> Optional[str]:
        return self._as_of
