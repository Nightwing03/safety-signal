"""Deterministic checks on model output. Pure functions: text in, list of problems out (empty = pass)."""
import re

from safety_signal.facts import Facts
from safety_signal.text import contains_phrase, normalize

MAX_WORDS = 150

# (pattern, reason). Matched on lower-cased text. Deliberately blunt: a false alarm costs a retry or the
# template fallback; a miss puts an unsupported claim in front of a reader.
FORBIDDEN = [
    (r"\bcaus(e|es|ed|ing)\b", "implies causation"),
    (r"\bprov(e|es|en|ing)\b", "claims proof"),
    (r"\bconfirm(s|ed|ing)?\b", "claims confirmation"),
    (r"\b(unsafe|dangerous|harmful|toxic)\b", "makes a safety judgement"),
    (r"\b(is|are) safe\b", "makes a safety judgement"),
    (r"\b(incidence|prevalence|likelihood|probability)\b", "implies a rate FAERS cannot give"),
    (r"\b(dose|doses|dosage|dosing)\b", "mentions dosing"),
    (r"\b(you|patients|people|users) should\b", "gives advice"),
    (r"\b(stop taking|discontinue|avoid|switch to)\b", "gives advice"),
    (r"\b(recommend|advise|advice|consult)\w*", "gives advice"),
    (r"\bdiagnos\w*", "medical claim"),
    (r"\b(new|novel|unknown|undisclosed|undiscovered|previously unrecognized|hidden) (risk|risks|hazard|safety|adverse|side effect)", "overstates a candidate"),
    (r"\b(more|most|less) (dangerous|serious|severe|harmful)\b", "ranks severity"),
    (r"\bworst\b", "ranks severity"),
]
_FORBIDDEN = [(re.compile(p), why) for p, why in FORBIDDEN]
_NUM = re.compile(r"(?<![\d.,])\d[\d,]*(?:\.\d+)?%?")  # also catches digits glued to letters, e.g. "PRR2.04"


def allowed_numbers(facts: Facts) -> set:
    vals = {float(len(facts.findings)), float(facts.label_records), 2.0, 4.0, 3.0}  # 2, 4, 3: screening rule
    vals |= {float(v) for v in facts.counts().values()}
    for f in facts.findings:
        vals.add(float(f.reports))
        if f.prr is not None:
            vals.add(float(f"{f.prr:.2f}"))
        if f.chi2 is not None:
            vals.add(float(f"{f.chi2:.1f}"))
    if facts.as_of:
        vals |= {float(x) for x in re.findall(r"\d+", facts.as_of)}
    return vals


def _strip_names(text: str, facts: Facts) -> str:
    # Reaction names can contain digits ("TYPE 2 DIABETES MELLITUS"); those are not numeric claims.
    for f in facts.findings:
        text = re.sub(re.escape(f.reaction), " ", text, flags=re.I)
    return text


_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_NOT_FOUND = "not found in the sampled label text"
_FOUND = re.compile(r"(?<!not )found in the sampled label text")
_UNCHECKED = "not checked against a label"
_STATUS_PHRASE = {"labeled": "found", "candidate": "not_found", "no_label_data": "unchecked"}


def _status_phrases(sentence: str) -> set:
    low = sentence.lower()
    found = set()
    if _NOT_FOUND in low:
        found.add("not_found")
    if _FOUND.search(low):
        found.add("found")
    if _UNCHECKED in low:
        found.add("unchecked")
    return found


def binding_problems(text: str, facts: Facts, require_flagged: bool) -> list:
    """Each sentence that names reactions and states a label status must state the status those reactions have.

    Found by a live run: a model wrote "DIARRHOEA ... were not found in the sampled label text" for a labeled
    reaction and every other check passed, because the words and numbers were all individually legitimate."""
    problems, covered = [], set()
    for sentence in _SENTENCES.split(text):
        phrases = _status_phrases(sentence)
        norm = normalize(sentence)
        mentioned = [f for f in facts.findings if contains_phrase(norm, f.reaction)]
        # "PAIN" inside "PAIN IN EXTREMITY" is not a separate mention.
        mentioned = [
            f for f in mentioned
            if not any(g is not f and contains_phrase(normalize(g.reaction), f.reaction) and len(g.reaction) > len(f.reaction)
                       for g in mentioned)
        ]
        if not mentioned or not phrases:
            continue
        if len(phrases) > 1:
            problems.append("one sentence states different label statuses; use one sentence per label status")
            continue
        (phrase,) = phrases
        sentence_numbers = {
            float(t.replace(",", "")) for t in _NUM.findall(_strip_names(sentence, facts)) if not t.endswith("%")
        }
        for f in mentioned:
            if _STATUS_PHRASE[f.label_status] == phrase:
                covered.add(f.reaction)
                if require_flagged and f.flagged:
                    needed = [float(f.reports)] + ([float(f"{f.prr:.2f}")] if f.prr is not None else [])
                    if any(n not in sentence_numbers for n in needed):
                        problems.append(f"the sentence about {f.reaction!r} must give its reports and PRR")
            else:
                problems.append(
                    f"{f.reaction!r} is described as '{phrase.replace('_', ' ')}' but its label_status is {f.label_status!r}"
                )
    if require_flagged:
        for f in facts.flagged:
            if f.reaction not in covered:
                problems.append(f"does not state the correct label status of flagged reaction {f.reaction!r}")
    return problems


def check_text(text: str, cited: list, facts: Facts, require_flagged: bool = True) -> list:
    problems = []
    if len(text.split()) > MAX_WORDS:
        problems.append(f"longer than {MAX_WORDS} words")
    allowed = allowed_numbers(facts)
    stripped = _strip_names(text, facts)
    for tok in _NUM.findall(stripped):
        if tok.endswith("%"):
            problems.append(f"percentage {tok!r} is not in the facts")
            continue
        try:
            value = float(tok.replace(",", ""))
        except ValueError:
            continue
        if value not in allowed:
            problems.append(f"number {tok!r} is not in the facts")
    low = text.lower()
    for pattern, why in _FORBIDDEN:
        m = pattern.search(stripped.lower())  # words inside reaction names (e.g. "DOSE OMISSION") are not claims
        if m:
            problems.append(f"forbidden wording {m.group(0)!r}: {why}")
    # Names are compared after spelling normalisation (DIARRHOEA == Diarrhea): same reaction, different spelling.
    known = {normalize(f.reaction) for f in facts.findings}
    for name in cited:
        if normalize(name) not in known:
            problems.append(f"cites a reaction that is not in the facts: {name!r}")
    if require_flagged:
        norm_text = normalize(text)
        for f in facts.flagged:
            if not contains_phrase(norm_text, f.reaction):
                problems.append(f"does not mention flagged reaction {f.reaction!r}")
    problems.extend(binding_problems(text, facts, require_flagged))
    return problems
