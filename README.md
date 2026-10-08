# safety-signal

Screens FDA adverse-event reports (openFDA FAERS) for a drug, flags reactions that are reported disproportionately often, checks each flagged reaction against the drug's FDA label text, and writes a short plain-language summary with a language model whose output is verified by code before anyone sees it.

**This is a screening tool, not medical advice.** A flag means "reported more often than for other drugs in this database". It does not show that the drug causes the reaction.

## What it does

```
drug name -> FAERS counts (openFDA) -> PRR / chi-square flags -> label comparison -> facts JSON
                                                                                       |
                                  header + caveat <- deterministic checks <- model summary (or template)
```

- **Signals.** For the top reactions of a drug it builds a 2x2 table and computes PRR and Pearson chi-square. A pair is flagged when PRR >= 2, chi-square >= 4 and at least 3 reports (the Evans rule).
- **Label check.** Each reaction is looked for in the drug's label sections (boxed warning, warnings, adverse reactions, contraindications) across up to 40 single-ingredient label records. Result per reaction: `labeled`, `candidate` (not found in the sampled label text) or `no_label_data`.
- **Summary.** The model sees only the computed facts as JSON, never raw data. Code adds the header and the caveat, so the model cannot remove them.
- **Checks on the model's text.** Every number must appear in the facts; no percentages; no causal or confirmatory wording; reaction names must be real ones; each sentence must state the correct label status for the reactions it names; every flagged reaction must be mentioned with its report count and PRR; 150-word cap. One retry with the specific problems. If it still fails, a deterministic template is used. When nothing is flagged the model is not called.
- **Questions.** `POST /ask` refuses medical-advice, dosing and causation questions (regex prefilter plus a model sentinel) and otherwise answers only from the facts.

## Run it

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q                                  # tests use fakes; no network, no keys
python -m safety_signal metformin --top 15           # table with label column
python -m safety_signal metformin --summary          # adds the checked summary
uvicorn safety_signal.api:default_app --factory --port 8000
```

Optional keys go in `.env` (gitignored): `OPENFDA_API_KEY` (raises the openFDA daily limit), `GROQ_API_KEY`, `GROQ_MODEL` (default `openai/gpt-oss-20b`). Without a model key the summary falls back to a local Ollama model if present, and otherwise to the template.

The model client comes from a sibling repository, `llm-service-template` (provider fallback, retries, rate limiting). Clone it next to this repo or set `LLM_SERVICE_PATH`. Everything except live summaries works without it.

API: `GET /health`, `GET /signals/{drug}`, `GET /summary/{drug}`, `POST /ask` with `{"drug": "...", "question": "..."}`. Responses include which provider and model answered and how many attempts were needed.

## How well does it work (measured)

**Label matching** (72 reaction-drug pairs across 6 drugs, human-labelled "is this reaction named in the label text"). Dev drugs were used to tune the matcher; holdout drugs were not.

| | precision | recall |
|---|---|---|
| dev, first run | 0.81 | 0.87 |
| dev, after tuning | 1.00 | 0.93 |
| holdout, first run | 0.77 | 0.96 |
| holdout, after tuning | 0.82 | 0.96 |

The honest holdout figure is the first-run one. The after-tuning holdout numbers are contaminated: I looked at holdout disagreements before making two matcher fixes (a standalone rule for the generic term "pain" and the removal of one alias). A baseline that calls everything "labeled" scores precision 0.60.

**Summary checks** (6 drugs x 5 repeats = 30 runs, `openai/gpt-oss-20b` via Groq): 29 passed on the first attempt, 1 passed after a retry (the model omitted a flagged reaction), 0 fell back to the template. In one run Groq failed and the local model answered. Summary call latency: median 1.1 to 2.2 s per drug, with occasional slow outliers (p95 up to 18 s). A cold run that fetches fresh openFDA data takes about 14 s; repeats are served from a 24-hour cache.

Raw runs: `measurements/summary_runs.jsonl`. Reproduce: `python -m safety_signal.measure metformin lisinopril --repeats 5`.

## Limits you should know about

- **FAERS is not incidence data.** Reports are voluntary, unvalidated, can be duplicated, and reflect what gets reported, not what happens. PRR can be inflated by indication (a drug for diabetes is reported with "blood glucose increased").
- **"Found in label" means the term appears in the sampled label text**, not that the label lists it as an adverse effect. The matcher cannot tell "causes X" from "not recommended in patients with X". "Not found" can also mean a synonym the matcher does not know, or that the sampled records differ.
- **The evaluation is small and has one labeller.** 72 rows, labels drafted with LLM assistance and every row reviewed by one person. The 0.82 vs 0.77 precision gap on the holdout is within what 72 rows can resolve; I would not claim an improvement from it.
- **The output checks are lexical and structural, not semantic.** They catch wrong numbers, wrong label status and forbidden wording. They cannot catch a fluent sentence that misleads in a way I did not write a check for. An earlier version of the checks passed a summary that gave the wrong label status; that is why the per-sentence status check exists.
- **30 runs on 6 drugs is a small sample.** The pass rate describes these inputs and this model on the day I ran it. Models change; Groq retired the model this project started on during development.
- **The refusal filter is unmeasured.** It is a regex prefilter plus a model sentinel; I have not built a labelled question set for it yet.
- **No authentication and no per-client rate limiting on the API.** Do not expose it publicly as is.

## Layout

`client.py` (HTTP, retries, rate limit, cache) | `stats.py` (PRR, chi-square) | `events.py`, `labels.py` (data sources behind interfaces) | `pairs.py`, `matcher.py`, `text.py` | `facts.py` | `summary.py`, `checks.py`, `llm.py` | `api.py`, `cli.py` | `evaluation.py` (labelling and scoring CLI) | `measure.py` | `eval/` (labelled pairs) | `tests/`
