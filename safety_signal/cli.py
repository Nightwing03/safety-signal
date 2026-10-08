"""Command line: python -m safety_signal metformin --top 15"""
import argparse
import sys
from typing import Callable, Optional, TextIO

from safety_signal.cache import NullCache, SqliteCache
from safety_signal.client import OpenFDAClient
from safety_signal.config import load_api_key
from safety_signal.errors import SafetySignalError
from safety_signal.events import EventSource, OpenFDAEvents
from safety_signal.pairs import PairResult, analyze_drug

CAVEAT = (
    "FAERS reports are voluntary and unvalidated. A flag is a screening result, not evidence of "
    "causation or incidence. Informational only, not medical advice."
)


def format_results(drug: str, results: list[PairResult], as_of: Optional[str]) -> str:
    lines = [f"Drug: {drug}    data as of: {as_of or 'unknown'}", CAVEAT, ""]
    if not results:
        lines.append("No reactions found for this drug.")
        return "\n".join(lines)
    lines.append(f"{'Reaction':<44}{'Reports':>9}{'PRR':>8}{'Chi2':>11}  Flag")
    for r in results:
        s = r.signal
        prr = "n/a" if s.prr is None else f"{s.prr:.2f}"
        chi2 = "n/a" if s.chi2 is None else f"{s.chi2:.1f}"
        lines.append(
            f"{r.reaction[:43]:<44}{s.table.a:>9}{prr:>8}{chi2:>11}  {'SIGNAL' if s.flagged else ''}"
        )
    return "\n".join(lines)


def default_source(args: argparse.Namespace) -> EventSource:
    key = load_api_key()
    if key is None:
        print("warning: OPENFDA_API_KEY not set; limited to 1,000 requests/day.", file=sys.stderr)
    cache = NullCache() if args.no_cache else SqliteCache(args.cache)
    return OpenFDAEvents(OpenFDAClient(api_key=key, cache=cache))


def main(
    argv: Optional[list[str]] = None,
    source_factory: Callable[[argparse.Namespace], EventSource] = default_source,
    out: Optional[TextIO] = None,
) -> int:
    out = out if out is not None else sys.stdout
    parser = argparse.ArgumentParser(prog="safety_signal", description="Screen drug-reaction pairs in FAERS.")
    parser.add_argument("drugs", nargs="+", help="generic drug names, e.g. metformin")
    parser.add_argument("--top", type=int, default=15, help="reactions to analyse per drug")
    parser.add_argument("--cache", default=".cache/openfda.sqlite")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = source_factory(args)
        for drug in args.drugs:
            results = analyze_drug(source, drug, top_n=args.top)
            print(format_results(drug, results, source.as_of()), file=out)
            print(file=out)
    except SafetySignalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
