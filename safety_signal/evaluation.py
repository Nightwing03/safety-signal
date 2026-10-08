"""Measures the label matcher against human judgement.

prepare: for each drug, pick one reference label, list the top reactions, and write a CSV for a person to fill in.
         The CSV deliberately does NOT contain the matcher's answer.
score:   read the filled CSVs, run the matcher on the same reference label, report precision and recall.

Human question for each row: "Does this label state the reaction as an effect of the drug
(adverse reaction, warning, or precaution)? A mention only as a risk factor or as the reason for use is NO."
"""
import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from safety_signal.labels import Label
from safety_signal.matcher import LabelIndex
from safety_signal.pairs import analyze_drug
from safety_signal.text import normalize

STOP_WORDS = {"increased", "decreased", "abnormal", "of", "and", "the", "blood", "with", "due"}
COLUMNS = ["reference_set_id", "drug", "reaction", "human", "notes", "snippets"]


def stems(reaction: str, k: int = 2) -> list[str]:
    """Broad search stems (first 5 letters of the longest words). Independent of the matcher's alias table."""
    tokens = [t for t in normalize(reaction).split() if t not in STOP_WORDS]
    tokens = sorted(tokens, key=len, reverse=True)[:k]
    return [t[:5] if len(t) > 5 else t for t in tokens]


def snippets(sections: dict, reaction: str, width: int = 80, max_n: int = 3) -> list[str]:
    """Context windows around broad-stem hits, so a person can judge without reading the whole label."""
    out: list[str] = []
    for name, text in sections.items():
        low = text.lower()
        last_start = -10**9
        for stem in stems(reaction):
            for m in re.finditer(re.escape(stem), low):
                start = max(0, m.start() - width)
                if abs(start - last_start) < width:
                    continue
                last_start = start
                out.append(f"[{name}] " + text[start : min(len(text), m.end() + width)].replace("\n", " "))
                if len(out) >= max_n:
                    return out
    return out


def pick_reference(labels: list[Label]) -> Label:
    """Deterministic: lowest set_id."""
    return min(labels, key=lambda lab: lab.set_id)


@dataclass(frozen=True)
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> Optional[float]:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> Optional[float]:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None


def score_pairs(pairs: list[tuple[bool, bool]]) -> Metrics:
    """pairs of (human says labeled, matcher says labeled)."""
    tp = sum(1 for h, m in pairs if h and m)
    fp = sum(1 for h, m in pairs if not h and m)
    fn = sum(1 for h, m in pairs if h and not m)
    tn = sum(1 for h, m in pairs if not h and not m)
    return Metrics(tp, fp, fn, tn)


def prepare(drugs, top_n, out_dir, source, label_source, out=sys.stdout) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for drug in drugs:
        labels = label_source.labels(drug)
        results = analyze_drug(source, drug, top_n=top_n)
        if not labels or not results:
            print(f"skipped {drug}: no label records or no reactions", file=out)
            continue
        ref = pick_reference(labels)
        path = out_dir / f"{drug}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in results:
                w.writerow(
                    {
                        "reference_set_id": ref.set_id,
                        "drug": drug,
                        "reaction": r.reaction,
                        "human": "",
                        "notes": "",
                        "snippets": " || ".join(snippets(ref.sections, r.reaction)),
                    }
                )
        (out_dir / f"{drug}_label.txt").write_text(
            "\n\n".join(f"## {name}\n{text}" for name, text in ref.sections.items())
        )
        written.append(path)
        print(f"wrote {path} ({len(results)} rows) and {drug}_label.txt", file=out)
    return written


def score(paths, holdout, label_source):
    """Returns (metrics by group, disagreements, unlabeled row count)."""
    holdout = {h.lower() for h in holdout}
    groups: dict[str, list] = {"dev": [], "holdout": []}
    misses = []
    unlabeled = 0
    for path in paths:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        drug, set_id = rows[0]["drug"], rows[0]["reference_set_id"]
        ref = next((lab for lab in label_source.labels(drug) if lab.set_id == set_id), None)
        if ref is None:
            raise SystemExit(f"reference label {set_id} for {drug} not found; re-run prepare")
        index = LabelIndex([ref])
        for row in rows:
            answer = row["human"].strip().lower()
            if answer not in ("y", "n"):
                unlabeled += 1
                continue
            human = answer == "y"
            matcher = index.match(row["reaction"]).status() == "labeled"
            groups["holdout" if drug.lower() in holdout else "dev"].append((human, matcher))
            if human != matcher:
                misses.append((drug, row["reaction"], "human=yes matcher=no" if human else "human=no matcher=yes", row["notes"]))
    return {k: score_pairs(v) for k, v in groups.items()}, misses, unlabeled


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def format_report(metrics, misses, unlabeled) -> str:
    lines = []
    for group, m in metrics.items():
        lines.append(
            f"{group:<8} n={m.n:<4} precision={_fmt(m.precision)} recall={_fmt(m.recall)}  "
            f"(tp={m.tp} fp={m.fp} fn={m.fn} tn={m.tn})"
        )
    if unlabeled:
        lines.append(f"{unlabeled} rows had no y/n answer and were skipped")
    if misses:
        lines.append("")
        lines.append("Disagreements:")
        for drug, reaction, kind, notes in misses:
            lines.append(f"  {drug:<14}{reaction:<40}{kind}  {notes}")
    return "\n".join(lines)


def main(argv=None, source_factory=None, label_factory=None, out=None) -> int:
    from safety_signal.cli import default_labels, default_source

    source_factory = source_factory or default_source
    label_factory = label_factory or default_labels
    out = out if out is not None else sys.stdout
    parser = argparse.ArgumentParser(prog="safety_signal.evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("drugs", nargs="+")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out", default="eval")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--holdout", nargs="*", default=[])
    for sp in (p, s):
        sp.add_argument("--cache", default=".cache/openfda.sqlite")
        sp.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.drugs, args.top, args.out, source_factory(args), label_factory(args), out=out)
    else:
        metrics, misses, unlabeled = score(args.files, args.holdout, label_factory(args))
        print(format_report(metrics, misses, unlabeled), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
