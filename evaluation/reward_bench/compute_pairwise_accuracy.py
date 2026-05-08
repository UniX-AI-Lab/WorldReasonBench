#!/usr/bin/env python3
"""Compute pairwise accuracy with and without ties from a JSONL file.

Usage:
    python3 compute_pairwise_accuracy.py <jsonl_path> [--pairs-json <gold_path>]

If --pairs-json is given, gold labels are loaded from there (for rescore).
Otherwise uses the expected_verdict already in the JSONL rows.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_dedup(path: str) -> list:
    by_id = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                pid = r.get("pair_id")
                if pid:
                    by_id[pid] = r
            except json.JSONDecodeError:
                continue
    return list(by_id.values())


def get_expected(pair):
    s1, s2 = float(pair.get("score_1", 0)), float(pair.get("score_2", 0))
    if pair.get("pair_type") == "tie" or abs(s1 - s2) < 1e-9:
        return "A=B"
    return "A>B" if s1 > s2 else "B>A"


def is_correct(parsed_verdict, expected_verdict):
    if parsed_verdict is None:
        return None
    if expected_verdict == "A=B":
        return parsed_verdict in ("A=B=Good", "A=B=Bad")
    return parsed_verdict == expected_verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="Path to pairwise eval JSONL")
    parser.add_argument("--pairs-json", default=None,
                        help="Optional gold pairs JSON for rescore")
    args = parser.parse_args()

    rows = load_dedup(args.jsonl)

    # Optionally rescore against gold
    if args.pairs_json:
        gold_pairs = json.load(open(args.pairs_json))
        gold_by_pid = {}
        for p in gold_pairs:
            pid = f"{p['task_id']}::{p['model_1']}::{p['model_2']}"
            gold_by_pid[pid] = p
        for r in rows:
            g = gold_by_pid.get(r.get("pair_id"))
            if g:
                r["expected_verdict"] = get_expected(g)
                r["is_correct"] = is_correct(r.get("parsed_verdict"), r["expected_verdict"])

    parsed = [r for r in rows if r.get("parsed_verdict")]
    print(f"Total rows: {len(rows)}")
    print(f"Parsed: {len(parsed)} ({len(parsed)/len(rows)*100:.1f}%)")
    print()

    # Overall
    all_correct = sum(1 for r in parsed if r.get("is_correct"))
    gap_parsed = [r for r in parsed if r.get("expected_verdict") != "A=B"]
    gap_correct = sum(1 for r in gap_parsed if r.get("is_correct"))
    tie_parsed = [r for r in parsed if r.get("expected_verdict") == "A=B"]
    tie_correct = sum(1 for r in tie_parsed if r.get("is_correct"))

    print(f"{'':35s} {'w/ Ties':>10s}  {'w/o Ties':>10s}  {'Tie only':>10s}")
    print("-" * 70)
    print(f"{'Overall':35s} {all_correct/len(parsed)*100:9.2f}%  "
          f"{gap_correct/len(gap_parsed)*100 if gap_parsed else 0:9.2f}%  "
          f"{tie_correct/len(tie_parsed)*100 if tie_parsed else 0:9.2f}%")

    # Per category
    cats = {}
    for r in parsed:
        c = r.get("category", "?")
        cats.setdefault(c, {"all": [0, 0], "gap": [0, 0], "tie": [0, 0]})
        cats[c]["all"][0] += 1
        if r.get("is_correct"):
            cats[c]["all"][1] += 1
        if r.get("expected_verdict") != "A=B":
            cats[c]["gap"][0] += 1
            if r.get("is_correct"):
                cats[c]["gap"][1] += 1
        else:
            cats[c]["tie"][0] += 1
            if r.get("is_correct"):
                cats[c]["tie"][1] += 1

    for c in sorted(cats):
        d = cats[c]
        a1 = d["all"][1] / d["all"][0] * 100 if d["all"][0] else 0
        a2 = d["gap"][1] / d["gap"][0] * 100 if d["gap"][0] else 0
        a3 = d["tie"][1] / d["tie"][0] * 100 if d["tie"][0] else 0
        print(f"{c:35s} {a1:9.2f}%  {a2:9.2f}%  {a3:9.2f}%")

    # Verdict distribution
    print()
    pred = Counter(r.get("parsed_verdict") for r in parsed)
    exp = Counter(r.get("expected_verdict") for r in parsed)
    print(f"Predicted verdicts: {dict(pred)}")
    print(f"Expected verdicts:  {dict(exp)}")


if __name__ == "__main__":
    main()
