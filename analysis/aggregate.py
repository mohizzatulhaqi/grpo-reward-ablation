"""Aggregate per-run eval JSONs into a comparison table against the baseline.

Each run writes results/{variant}_seed{seed}_{dataset}.json containing at least
{"variant","seed","dataset","pass@1", ...}. `build_table` is pure/in-memory so
it is unit-tested; `main` just handles file IO.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from analysis.stats import compare


def build_table(records, metric="pass@1", baseline="A0", paired=True):
    """records: list of dicts with keys 'variant','seed',metric.
    Returns a list of row dicts (one per variant), each compared to `baseline`.
    """
    by_variant = defaultdict(dict)  # variant -> {seed: value}
    for r in records:
        if metric in r:
            by_variant[r["variant"]][r["seed"]] = r[metric]

    if baseline not in by_variant:
        raise ValueError(f"baseline {baseline!r} not found in records")

    base_seeds = sorted(by_variant[baseline])
    base_vals = [by_variant[baseline][s] for s in base_seeds]

    rows = []
    for variant in sorted(by_variant):
        seeds = sorted(by_variant[variant])
        vals = [by_variant[variant][s] for s in seeds]
        row = {"variant": variant, "n": len(vals),
               "mean": sum(vals) / len(vals) if vals else float("nan")}
        if variant == baseline:
            row.update({"delta_vs_base": 0.0, "p_value": float("nan"),
                        "cliffs": 0.0, "effect": "-"})
        else:
            # pair on common seeds when paired=True
            common = sorted(set(seeds) & set(base_seeds)) if paired else None
            if paired and common:
                a = [by_variant[variant][s] for s in common]
                b = [by_variant[baseline][s] for s in common]
            else:
                a, b = vals, base_vals
            c = compare(a, b, paired=paired and bool(common))
            row.update({
                "delta_vs_base": row["mean"] - (sum(base_vals) / len(base_vals)),
                "p_value": c["p_value"],
                "cliffs": c["cliffs_delta"],
                "effect": c["effect"],
            })
        rows.append(row)
    return rows


def render_markdown(rows, metric="pass@1", baseline="A0"):
    head = (f"| variant | n | {metric} (mean) | Δ vs {baseline} | "
            f"p (vs {baseline}) | Cliff's δ | effect |")
    sep = "|" + "|".join(["---"] * 7) + "|"
    lines = [head, sep]
    for r in rows:
        p = "—" if r["p_value"] != r["p_value"] else f"{r['p_value']:.4f}"
        lines.append(
            f"| {r['variant']} | {r['n']} | {r['mean']:.4f} | "
            f"{r['delta_vs_base']:+.4f} | {p} | {r['cliffs']:+.3f} | {r['effect']} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--metric", default="pass@1")
    ap.add_argument("--baseline", default="A0")
    ap.add_argument("--dataset", default="gsm8k")
    args = ap.parse_args()

    records = []
    for path in glob.glob(os.path.join(args.results, "*.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("dataset", args.dataset) == args.dataset:
            records.append(r)

    if not records:
        print(f"No result JSONs found in {args.results}/ for dataset={args.dataset}")
        return
    rows = build_table(records, metric=args.metric, baseline=args.baseline)
    print(f"\nDataset: {args.dataset} | metric: {args.metric}\n")
    print(render_markdown(rows, metric=args.metric, baseline=args.baseline))


if __name__ == "__main__":
    main()
