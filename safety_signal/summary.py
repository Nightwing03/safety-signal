"""Grounded summary and Q&A. The model writes prose; code checks it; if it cannot be made to pass, a
deterministic template is used instead, so a reader never sees unchecked model text."""
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel

from safety_signal.checks import check_text
from safety_signal.facts import Facts

CAVEAT = (
    "FAERS reports are voluntary and unvalidated. A flag is a screening result, not evidence of causation "
    "or incidence. A reaction marked 'not found in label' means the term was not found in the sampled label "
    "text, not that it is unknown. Informational only, not medical advice."
)
OUT_OF_SCOPE = "OUT_OF_SCOPE"
REFUSAL = (
    "I can only describe the screening results shown here. I can't give medical advice, dosing, safety "
    "judgements or causal explanations. Please ask a pharmacist or clinician."
)

SUMMARY_SYSTEM = """You write a short plain-language summary of a drug-safety screening result for a technical reader.
Rules:
- Use only the facts in the JSON.
- The JSON has "flagged_by_label_status": three lists of reaction names, already grouped. Write one sentence for each NON-EMPTY list, in this order: "labeled", "candidate", "no_label_data".
- Each sentence names only the reactions in its own list, copying each name exactly as written (keep its spelling: DIARRHOEA stays DIARRHOEA), with that reaction's reports and PRR taken from the "reactions" array.
- Sentence endings by list: "labeled" -> "found in the sampled label text"; "candidate" -> "not found in the sampled label text"; "no_label_data" -> "not checked against a label".
- After each reaction name, give in parentheses its reports and PRR from the "reactions" array, as in "AAA (500 reports, PRR 3.10)".
- Copy numbers exactly as they appear in the JSON. Do not compute, round or add numbers.
- Begin with one sentence giving the number of reactions analysed and the number flagged (from "counts").
- Describe only what the numbers and label_status show. Add no interpretation, no commentary on what the results mean, no comparison with other drugs, and no advice.
- At most 120 words, plain sentences, no markdown.
Example of the format (placeholder names and numbers):
{"summary": "Of 10 reactions analysed, 3 were flagged. AAA (500 reports, PRR 3.10) and CCC (200 reports, PRR 2.20) were found in the sampled label text. BBB (300 reports, PRR 2.40) was not found in the sampled label text.", "cited_reactions": ["AAA", "CCC", "BBB"]}
Return JSON in exactly that shape."""

ANSWER_SYSTEM = f"""You answer a question using ONLY the screening facts in the JSON.
- If the question is not answerable from the facts, or asks for medical advice, dosing, safety judgements, causation or anything about a different drug, set "answer" to exactly "{OUT_OF_SCOPE}".
- Quote numbers exactly as in the JSON. No outside knowledge. No causation, incidence, severity ranking or advice.
- At most 80 words, plain sentences.
Return JSON: {{"answer": "...", "cited_reactions": ["EXACT NAME", ...]}}"""


class SummaryOut(BaseModel):
    summary: str
    cited_reactions: list[str] = []


class AnswerOut(BaseModel):
    answer: str
    cited_reactions: list[str] = []


@dataclass
class SummaryResult:
    text: str
    source: str  # "llm" | "template" | "refusal"
    attempts: int = 0
    problems: list = field(default_factory=list)  # one list of problems per failed attempt
    error: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    providers_tried: list = field(default_factory=list)  # providers that failed before the one that answered


def _add(a, b):
    return b if a is None else (a if b is None else a + b)


def _fold(result: SummaryResult, res: Any) -> None:
    result.provider = getattr(res, "provider", result.provider)
    result.model = getattr(res, "model", result.model)
    tried = getattr(res, "providers_tried", None)
    result.providers_tried = list(tried) if isinstance(tried, (list, tuple)) else result.providers_tried
    result.latency_ms += float(getattr(res, "latency_ms", 0.0) or 0.0)
    result.input_tokens = _add(result.input_tokens, getattr(res, "input_tokens", None))
    result.output_tokens = _add(result.output_tokens, getattr(res, "output_tokens", None))
    result.cost_usd = _add(result.cost_usd, getattr(res, "cost_usd", None))


def render(facts: Facts, body: str) -> str:
    header = f"{facts.drug}: screening summary (data as of {facts.as_of or 'unknown'})"
    return f"{header}\n{body.strip()}\n\n{CAVEAT}"


_STATUS_WORDS = {
    "labeled": "found in the sampled label text",
    "candidate": "not found in the sampled label text",
    "no_label_data": "not checked against a label (no label records)",
}


def template_body(facts: Facts) -> str:
    c = facts.counts()
    if not facts.flagged:
        return (
            f"Of {c['reactions_analysed']} reactions analysed, none met the screening rule "
            "(PRR of at least 2, chi-square of at least 4, at least 3 reports)."
        )
    lines = [
        f"Of {c['reactions_analysed']} reactions analysed, {c['flagged']} met the screening rule "
        "(PRR of at least 2, chi-square of at least 4, at least 3 reports):"
    ]
    for f in facts.flagged:
        lines.append(
            f"- {f.reaction}: {f.reports} reports, PRR {f.prr:.2f}, chi-square {f.chi2:.1f}; "
            f"{_STATUS_WORDS[f.label_status]}."
        )
    return "\n".join(lines)


def _user_prompt(facts: Facts, previous_problems: Optional[list], question: Optional[str] = None) -> str:
    parts = ["FACTS (JSON):", facts.to_json()]
    if question is not None:
        parts += ["", f"QUESTION: {question}"]
    if previous_problems:
        parts += ["", "Your previous answer failed these checks. Fix them and answer again:"]
        parts += [f"- {p}" for p in previous_problems]
    return "\n".join(parts)


def summarize(facts: Facts, llm: Any, max_attempts: int = 2) -> SummaryResult:
    result = SummaryResult(text="", source="template")
    if not facts.flagged:  # nothing to explain; skip the model entirely
        result.text = render(facts, template_body(facts))
        return result
    previous = None
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        try:
            res = llm.complete(system=SUMMARY_SYSTEM, user=_user_prompt(facts, previous), response_model=SummaryOut)
        except Exception as exc:  # noqa: BLE001 - boundary: any provider failure means "use the template"
            result.error = type(exc).__name__
            break
        _fold(result, res)
        out = res.value
        problems = check_text(out.summary, out.cited_reactions, facts)
        if not problems:
            result.text, result.source = render(facts, out.summary), "llm"
            return result
        result.problems.append(problems)
        previous = problems
    result.text = render(facts, template_body(facts))
    return result


# Questions we never send to a model. Blunt on purpose; the model is also told to refuse.
_OUT_OF_SCOPE = [
    r"\bshould i\b", r"\bcan i\b", r"\bam i\b", r"\bis it (safe|ok|okay|bad|dangerous)\b",
    r"\b(stop|start|quit|skip|double|increase|decrease|reduce) (taking|using|my)\b", r"\b(dose|dosage|dosing|mg)\b",
    r"\b(pregnan\w*|breastfe\w*|nursing|child|children|kid)\b", r"\b(interact\w*|together with|combine\w*|mix\w*)\b",
    r"\b(cause|causes|caused|why does|why do)\b", r"\b(diagnos\w*|treat\w*|cure\w*|symptom\w*)\b",
    r"\b(better|worse|safer|best|worst) (than|drug|option)\b",
]
_OUT_OF_SCOPE = [re.compile(p, re.I) for p in _OUT_OF_SCOPE]


def out_of_scope(question: str) -> bool:
    return any(p.search(question) for p in _OUT_OF_SCOPE)


def answer_question(question: str, facts: Facts, llm: Any, max_attempts: int = 2) -> SummaryResult:
    if not question.strip() or out_of_scope(question):
        return SummaryResult(text=REFUSAL, source="refusal")
    result = SummaryResult(text="", source="refusal")
    previous = None
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        try:
            res = llm.complete(
                system=ANSWER_SYSTEM, user=_user_prompt(facts, previous, question), response_model=AnswerOut
            )
        except Exception as exc:  # noqa: BLE001
            result.error = type(exc).__name__
            break
        _fold(result, res)
        out = res.value
        if out.answer.strip() == OUT_OF_SCOPE:
            result.text = REFUSAL
            return result
        problems = check_text(out.answer, out.cited_reactions, facts, require_flagged=False)
        if not problems:
            result.text, result.source = out.answer.strip(), "llm"
            return result
        result.problems.append(problems)
        previous = problems
    result.text = "I couldn't produce an answer I can verify against the screening results."
    return result
