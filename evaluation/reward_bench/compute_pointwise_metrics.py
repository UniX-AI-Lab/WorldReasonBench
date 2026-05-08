#!/usr/bin/env python3
"""Compute point-wise evaluation metrics from video-level and induced-pair JSONL files.

Metrics:
  1. Induced pairwise accuracy (from induced_pairs JSONL)
  2. Spearman rank correlation between AI scores and human ratings (from video JSONL)

Usage:
    python3 compute_pointwise_metrics.py \
        --videos <video_level.jsonl> \
        --induced-pairs <induced_pairs.jsonl>

Either or both arguments can be provided.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_dedup(path: str, id_key: str = "pair_id") -> list:
    by_id = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rid = r.get(id_key)
                if rid:
                    by_id[rid] = r
            except json.JSONDecodeError:
                continue
    return list(by_id.values())


def spearman(xs, ys):
    """Compute Spearman rank correlation."""
    n = len(xs)
    if n < 3:
        return None

    def rank(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx * sy > 0 else 0.0


def correlation(xs, ys):
    """Compute Pearson correlation."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs) / n)
    sy = math.sqrt(sum((b - my) ** 2 for b in ys) / n)
    return cov / (sx * sy) if sx * sy > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", default=None,
                        help="Video-level JSONL with parsed_scores and human_score")
    parser.add_argument("--induced-pairs", default=None,
                        help="Induced pairwise JSONL with parsed_verdict and is_correct")
    args = parser.parse_args()

    if not args.videos and not args.induced_pairs:
        parser.error("At least one of --videos or --induced-pairs is required.")

    # === Induced pairwise accuracy ===
    if args.induced_pairs:
        rows = load_dedup(args.induced_pairs, "pair_id")
        parsed = [r for r in rows if r.get("parsed_verdict")]
        correct = sum(1 for r in parsed if r.get("is_correct"))
        print("=" * 60)
        print("INDUCED PAIRWISE ACCURACY")
        print("=" * 60)
        print(f"Total pairs: {len(rows)}")
        print(f"Parsed: {len(parsed)} ({len(parsed)/len(rows)*100:.1f}%)")
        print(f"Correct: {correct}")
        print(f"Accuracy: {correct/len(parsed)*100:.2f}%" if parsed else "N/A")
        print()
        print(f"{'Category':35s} {'Parsed':>8s}  {'Correct':>8s}  {'Acc':>8s}")
        print("-" * 65)
        cats = {}
        for r in parsed:
            c = r.get("category", "?")
            cats.setdefault(c, [0, 0])
            cats[c][0] += 1
            if r.get("is_correct"):
                cats[c][1] += 1
        for c in sorted(cats):
            t, cr = cats[c]
            print(f"{c:35s} {t:8d}  {cr:8d}  {cr/t*100:7.2f}%")
        total_p = sum(v[0] for v in cats.values())
        total_c = sum(v[1] for v in cats.values())
        print(f"{'Overall':35s} {total_p:8d}  {total_c:8d}  {total_c/total_p*100:7.2f}%")
        print()

    # === Spearman / Pearson correlation ===
    if args.videos:
        rows = load_dedup(args.videos, "video_id")
        parsed = [r for r in rows
                  if r.get("parsed_scores") and r.get("human_score") is not None
                  and r.get("score_weighted") is not None]
        print("=" * 60)
        print("SCORE CORRELATION (AI vs Human)")
        print("=" * 60)
        print(f"Total videos: {len(rows)}")
        print(f"Parsed with human_score: {len(parsed)}")
        print()

        if parsed:
            ai = [r["score_weighted"] for r in parsed]
            human = [float(r["human_score"]) for r in parsed]
            rho = spearman(ai, human)
            pear = correlation(ai, human)
            print(f"{'':35s} {'Spearman':>10s}  {'Pearson':>10s}  {'N':>6s}")
            print("-" * 65)

            cats = {}
            for r in parsed:
                c = r.get("category", "?")
                cats.setdefault(c, {"ai": [], "human": []})
                cats[c]["ai"].append(r["score_weighted"])
                cats[c]["human"].append(float(r["human_score"]))

            for c in sorted(cats):
                d = cats[c]
                rho_c = spearman(d["ai"], d["human"])
                pear_c = correlation(d["ai"], d["human"])
                n_c = len(d["ai"])
                rho_s = f"{rho_c:.4f}" if rho_c is not None else "N/A"
                pear_s = f"{pear_c:.4f}" if pear_c is not None else "N/A"
                print(f"{c:35s} {rho_s:>10s}  {pear_s:>10s}  {n_c:6d}")

            rho_s = f"{rho:.4f}" if rho is not None else "N/A"
            pear_s = f"{pear:.4f}" if pear is not None else "N/A"
            print(f"{'Overall':35s} {rho_s:>10s}  {pear_s:>10s}  {len(parsed):6d}")

            # Score distribution summary
            print()
            print("Score distribution (AI weighted):")
            print(f"  min={min(ai):.2f}  max={max(ai):.2f}  "
                  f"mean={sum(ai)/len(ai):.2f}  median={sorted(ai)[len(ai)//2]:.2f}")
            print("Score distribution (Human):")
            print(f"  min={min(human):.2f}  max={max(human):.2f}  "
                  f"mean={sum(human)/len(human):.2f}  median={sorted(human)[len(human)//2]:.2f}")

            # Inter-dimension correlation (halo effect check)
            scores_r = [r["parsed_scores"][0] for r in parsed]
            scores_c = [r["parsed_scores"][1] for r in parsed]
            scores_a = [r["parsed_scores"][2] for r in parsed]
            print()
            print("Inter-dimension correlation (halo effect check):")
            print(f"  reasoning-content:    Pearson={correlation(scores_r, scores_c):.3f}  "
                  f"Spearman={spearman(scores_r, scores_c):.3f}")
            print(f"  reasoning-aesthetics: Pearson={correlation(scores_r, scores_a):.3f}  "
                  f"Spearman={spearman(scores_r, scores_a):.3f}")
            print(f"  content-aesthetics:   Pearson={correlation(scores_c, scores_a):.3f}  "
                  f"Spearman={spearman(scores_c, scores_a):.3f}")


if __name__ == "__main__":
    main()
