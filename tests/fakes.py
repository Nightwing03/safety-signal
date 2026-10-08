"""Test doubles. Nothing here touches the network, the disk or real time."""


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class FakeHttp:
    """Plays back queued responses (or raises queued exceptions) and records every call."""

    def __init__(self, responses, clock=None):
        self.responses = list(responses)
        self.calls = []
        self.times = []
        self._clock = clock

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        if self._clock is not None:
            self.times.append(self._clock.now())
        if not self.responses:
            raise AssertionError("unexpected extra request")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    """Stands in for OpenFDAClient: handler(endpoint, params) returns a body or raises."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, endpoint, params):
        self.calls.append((endpoint, dict(params)))
        return self.handler(endpoint, params)


def total_body(total, last_updated="2026-01-01"):
    return {
        "meta": {"last_updated": last_updated, "results": {"skip": 0, "limit": 1, "total": total}},
        "results": [{}],
    }


class FakeEvents:
    """In-memory EventSource."""

    def __init__(self, n_total, drug_totals, reaction_totals, pairs, top, as_of="2026-01-01"):
        self.n_total = n_total
        self.drug_totals = drug_totals
        self.reaction_totals = reaction_totals
        self.pairs = pairs
        self.top = top
        self._as_of = as_of

    def total(self):
        return self.n_total

    def drug_total(self, drug):
        return self.drug_totals[drug]

    def reaction_total(self, reaction):
        return self.reaction_totals[reaction]

    def pair_count(self, drug, reaction):
        return self.pairs[(drug, reaction)]

    def top_reactions(self, drug):
        return list(self.top.get(drug, []))

    def as_of(self):
        return self._as_of


def make_events():
    """One drug with a clear signal, a null reaction, a stoplisted term and an undefined-PRR reaction."""
    return FakeEvents(
        n_total=100_000,
        drug_totals={"alpha": 1_000},
        reaction_totals={"RASH": 2_000, "NAUSEA": 10_000, "DRUG INEFFECTIVE": 5_000, "RARE": 2},
        pairs={
            ("alpha", "RASH"): 100,
            ("alpha", "NAUSEA"): 100,
            ("alpha", "DRUG INEFFECTIVE"): 500,
            ("alpha", "RARE"): 2,
        },
        top={"alpha": ["DRUG INEFFECTIVE", "NAUSEA", "RASH", "RARE"]},
    )


def make_label(set_id="x", generic="ALPHA", **sections):
    from safety_signal.labels import Label

    return Label(set_id, (generic,), (), None, dict(sections))


class FakeLabels:
    def __init__(self, labels):
        self._labels = list(labels)

    def labels(self, drug):
        return list(self._labels)


from types import SimpleNamespace  # noqa: E402

from safety_signal.facts import Facts, FindingFact  # noqa: E402


class FakeLLM:
    """Plays back queued outputs (or raises queued exceptions) and records each call."""

    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def complete(self, *, system=None, user, response_model=None, temperature=None):
        self.calls.append({"system": system, "user": user, "model": response_model})
        if not self.items:
            raise AssertionError("model was called but no response was queued")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            value=item, provider="fake", model="fake-1", latency_ms=10.0,
            input_tokens=100, output_tokens=50, cost_usd=0.0,
        )


def make_facts(drug="alpha", as_of="2026-07-30"):
    return Facts(
        drug, as_of, 12,
        (
            FindingFact("LACTIC ACIDOSIS", 19398, 72.89, 533800.6, True, "labeled", ("boxed_warning",)),
            FindingFact("ACUTE KIDNEY INJURY", 18251, 6.37, 73187.3, True, "candidate", ()),
            FindingFact("NAUSEA", 30165, 1.86, 11957.0, False, "labeled", ("adverse_reactions",)),
        ),
    )


GOOD_SUMMARY = (
    "Of 3 reactions analysed, 2 were flagged. LACTIC ACIDOSIS (19,398 reports, PRR 72.89) was found in the "
    "sampled label text. ACUTE KIDNEY INJURY (18,251 reports, PRR 6.37) was not found in the sampled label text."
)
