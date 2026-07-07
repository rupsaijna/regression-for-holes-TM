"""Distribution of spacing across all registered datasets.

For every dataset and every spacing definition (test points, standardized
feature space) this summarises the full distribution of spacing values:
count, mean, std, coefficient of variation, min / p25 / median / p75 / p90 /
p99 / max, skewness, and a normalized Shannon entropy (0 = all points equally
spaced/concentrated in one bin, 1 = spacing spread uniformly across its range --
i.e. how much the spacing varies / how much "spacing information" it carries).

No model training -- this is purely the geometry of the data, so it is fast.

Outputs:
  * outputs/spacing_distribution.csv
  * outputs/spacing_distribution.md
  * outputs/spacing_distribution/box_<spacing>.png   (datasets compared per spacing)

Usage:
  python spacing_distribution.py
  python spacing_distribution.py --bins 30 --datasets california ccpp energy
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import skew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from common import datasets, metrics
from common.data import get_split_data, standardize
from common import viz  # noqa: F401  (ensures Agg backend is set)
import matplotlib.pyplot as plt

SPACINGS = ["nn_to_train_k1", "knn_to_train_k5", "local_density_k5", "nn_to_test_k1"]


def norm_entropy(x, bins):
    """Shannon entropy of the histogram of x, normalized to [0, 1]."""
    counts, _ = np.histogram(x, bins=bins)
    p = counts[counts > 0] / counts.sum()
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log2(p)).sum() / np.log2(bins))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bins", type=int, default=20, help="histogram bins for the entropy measure")
    p.add_argument("--datasets", nargs="+", default=list(datasets.DATASETS))
    args = p.parse_args()

    out_dir = os.path.join(config.OUTPUTS_DIR, "spacing_distribution")
    os.makedirs(out_dir, exist_ok=True)

    # spacing arrays[spacing][dataset] -> values (for the box plots)
    by_spacing = {s: {} for s in SPACINGS}
    rows = []
    for key in args.datasets:
        try:
            Xtr, Xte, ytr, yte = get_split_data(key)
        except Exception as e:
            print(f"[skip] {key}: {type(e).__name__} ({str(e).splitlines()[0]})", flush=True)
            continue
        _, Xtr_s, Xte_s = standardize(Xtr, Xte)
        Xall_s = np.vstack([Xtr_s, Xte_s])
        sp = metrics.all_spacings(Xte_s, Xtr_s, Xall_s)
        for s in SPACINGS:
            v = sp[s]
            by_spacing[s][key] = v
            q = np.percentile(v, [25, 50, 75, 90, 99])
            mean = float(v.mean())
            rows.append(dict(
                dataset=key, n=len(v), spacing=s,
                mean=round(mean, 4), std=round(float(v.std()), 4),
                cv=round(float(v.std() / mean) if mean else float("nan"), 3),
                min=round(float(v.min()), 4), p25=round(float(q[0]), 4),
                median=round(float(q[1]), 4), p75=round(float(q[2]), 4),
                p90=round(float(q[3]), 4), p99=round(float(q[4]), 4),
                max=round(float(v.max()), 4),
                skew=round(float(skew(v)), 3),
                norm_entropy=round(norm_entropy(v, args.bins), 3),
            ))
        print(f"[done] {key}", flush=True)

    cols = ["dataset", "n", "spacing", "mean", "std", "cv", "min", "p25", "median",
            "p75", "p90", "p99", "max", "skew", "norm_entropy"]
    csv_path = os.path.join(config.OUTPUTS_DIR, "spacing_distribution.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    md_path = os.path.join(config.OUTPUTS_DIR, "spacing_distribution.md")
    with open(md_path, "w") as f:
        f.write("# Distribution of spacing across datasets (test points, standardized space)\n\n")
        f.write("`cv` = std/mean (relative spread). `skew` > 0 means a long tail of "
                "isolated points. `norm_entropy` in [0,1]: low = spacing concentrated "
                "(uniform geometry), high = spacing values vary widely.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")

    # box plots: one figure per spacing, datasets side by side (fliers hidden;
    # the table's p99/max/skew capture the tails)
    for s in SPACINGS:
        keys = [k for k in args.datasets if k in by_spacing[s]]
        data = [by_spacing[s][k] for k in keys]
        fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.1), 5))
        ax.boxplot(data, tick_labels=keys, showfliers=False)
        ax.set_title(f"Spacing distribution by dataset -- {s} (standardized, fliers hidden)")
        ax.set_ylabel("spacing")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = os.path.join(out_dir, f"box_{s}.png")
        fig.savefig(out, dpi=130); plt.close(fig)

    print(f"\nCSV -> {csv_path}\nMD  -> {md_path}\nfigures -> {out_dir}/box_*.png")
    # quick console view
    print(f"\n{'dataset':11s} {'spacing':17s} {'mean':>7s} {'cv':>6s} {'skew':>6s} {'n_ent':>6s}")
    for r in rows:
        print(f"{r['dataset']:11s} {r['spacing']:17s} {r['mean']:7} {r['cv']:6} {r['skew']:6} {r['norm_entropy']:6}")


if __name__ == "__main__":
    main()
