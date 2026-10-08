"""Measure how often the model's summary passes the checks.

python -m safety_signal.measure metformin lisinopril --repeats 5 --out measurements/summary_runs.jsonl

Facts are built once per drug (cached openFDA data), then the summary is requested `repeats` times. Each run is
one JSON line, so a later change to the checks or the model can be compared against the same input class.
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from typing import Callable, Optional

from safety_signal.summary import summarize


def run_repeats(facts, llm, repeats: int) -> list:
    rows = []
    for i in range(repeats):
        res = summarize(facts, llm)
        rows.append({
            "drug": facts.drug, "run": i + 1, "source": res.source, "attempts": res.attempts,
            "problems": res.problems, "error": res.error, "provider": res.provider, "model": res.model,
            "failed_providers": res.providers_tried, "latency_ms": round(res.latency_ms, 1),
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        })
    return rows


def _percentile(values: list, q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def aggregate(rows: list) -> dict:
    n = len(rows)
    if n == 0:
        return {"runs": 0}
    first = sum(1 for r in rows if r["source"] == "llm" and r["attempts"] == 1)
    retried = sum(1 for r in rows if r["source"] == "llm" and r["attempts"] > 1)
    fallback = sum(1 for r in rows if r["source"] == "template")
    lat = [r["latency_ms"] for r in rows]
    reasons = Counter(p for r in rows for attempt in r["problems"] for p in attempt)
    return {
        "runs": n, "first_attempt_pass": first, "passed_after_retry": retried, "template_fallback": fallback,
        "latency_median_ms": round(statistics.median(lat), 1), "latency_p95_ms": round(_percentile(lat, 0.95), 1),
        "providers": dict(Counter(f"{r['provider']}/{r['model']}" for r in rows)),
        "runs_with_failed_providers": sum(1 for r in rows if r["failed_providers"]),
        "top_failure_reasons": reasons.most_common(5),
    }


def main(argv: Optional[list] = None, build: Optional[Callable] = None, out=None) -> int:
    out = out if out is not None else sys.stdout
    p = argparse.ArgumentParser(prog="safety_signal.measure")
    p.add_argument("drugs", nargs="+")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--cache", default=".cache/openfda.sqlite")
    p.add_argument("--out", default="measurements/summary_runs.jsonl")
    args = p.parse_args(argv)
    if build is None:
        from safety_signal.cache import SqliteCache
        from safety_signal.client import OpenFDAClient
        from safety_signal.config import load_api_key
        from safety_signal.events import OpenFDAEvents
        from safety_signal.labels import OpenFDALabels
        from safety_signal.llm import build_llm
        from safety_signal.pipeline import analyse

        client = OpenFDAClient(api_key=load_api_key(), cache=SqliteCache(args.cache))
        source, labels, llm = OpenFDAEvents(client), OpenFDALabels(client), build_llm()
        build = lambda drug: (analyse(source, labels, drug, args.top), llm)  # noqa: E731
    all_rows = []
    for drug in args.drugs:
        facts, llm = build(drug)
        if facts is None:
            print(f"{drug}: no reports, skipped", file=out)
            continue
        rows = run_repeats(facts, llm, args.repeats)
        all_rows += rows
        a = aggregate(rows)
        print(f"{drug:<14} runs {a['runs']}  first-attempt {a['first_attempt_pass']}  retry {a['passed_after_retry']}  "
              f"fallback {a['template_fallback']}  median {a['latency_median_ms']:.0f} ms  p95 {a['latency_p95_ms']:.0f} ms", file=out)
    if all_rows:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as fh:
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
        total = aggregate(all_rows)
        print(f"\nOVERALL runs {total['runs']}  first-attempt {total['first_attempt_pass']}  retry "
              f"{total['passed_after_retry']}  fallback {total['template_fallback']}  "
              f"failed-provider runs {total['runs_with_failed_providers']}", file=out)
        print(f"models: {total['providers']}", file=out)
        for reason, count in total["top_failure_reasons"]:
            print(f"  {count}x {reason}", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
