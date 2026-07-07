"""Teacher-edge vs distillation-benefit correlation.

Question: does the teacher being a more accurate model (lower test RMSE than the unguided
30%-hole student) predict whether distilling from it HELPS the student? E10's manual check
suggested NO (energy = biggest edge, worst guidance) — this quantifies it.

For every per-dataset CSV in an experiment dir (they all carry rmse_unguided, rmse_M20 and
impr_<method>):
  teacher_edge = rmse_unguided - rmse_M20      (>0 = teacher is the more accurate model)
  benefit_<m>  = impr_<m> = rmse_unguided - rmse_<m>   (>0 = mode beats no-teacher)

Reports, across datasets (one point each), per method: Pearson r and Spearman rho between the
dataset's mean teacher_edge and its mean benefit, plus a scatter (datasets labelled). A robust
variant clips per-seed benefit to [-CLIP, CLIP] so the teacher_sign blow-ups don't dominate.

Usage:  python correlate_teacher_edge.py [--exp-dir outputs/distillation/exp10] [--clip 2.0]
Out:    <exp-dir>/teacher_edge_corr.{png,md,csv}
"""
import argparse, csv, glob, os, sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def _spearman(x, y):
    """Spearman rho = Pearson on average-ranks. Robust to the blow-up outliers."""
    def ranks(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float); r[order] = np.arange(len(a))
        # average ties
        a = np.asarray(a, float)
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def _pearson(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load(exp_dir):
    """Return {dataset: rows[]} for per-dataset CSVs (those with rmse_M20)."""
    data = {}
    for path in sorted(glob.glob(os.path.join(exp_dir, "*.csv"))):
        if path.endswith("_summary.csv"):
            continue
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows or "rmse_M20" not in rows[0] or "rmse_unguided" not in rows[0]:
            continue
        data[rows[0]["dataset"]] = rows
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=os.path.join(config.OUTPUTS_DIR, "distillation", "exp10"))
    ap.add_argument("--clip", type=float, default=2.0,
                    help="cap |benefit| per seed for the robust correlation (tames blow-ups)")
    args = ap.parse_args()

    data = load(args.exp_dir)
    if not data:
        sys.exit(f"No per-dataset CSVs (with rmse_M20) found in {args.exp_dir}")
    methods = [k[len("impr_"):] for k in next(iter(data.values()))[0] if k.startswith("impr_")]
    dss = list(data.keys())

    # per-dataset means
    edge = {}      # ds -> (mean, std) teacher edge
    benef = {ds: {} for ds in dss}          # ds -> method -> mean benefit (raw)
    benef_c = {ds: {} for ds in dss}        # ds -> method -> mean benefit (clipped)
    for ds, rows in data.items():
        e = np.array([float(r["rmse_unguided"]) - float(r["rmse_M20"]) for r in rows])
        edge[ds] = (float(e.mean()), float(e.std()))
        for m in methods:
            b = np.array([float(r[f"impr_{m}"]) for r in rows])
            benef[ds][m] = float(b.mean())
            benef_c[ds][m] = float(np.clip(b, -args.clip, args.clip).mean())

    # correlations across datasets (one point per dataset)
    ex = np.array([edge[ds][0] for ds in dss])
    summary = []
    for m in methods:
        by = np.array([benef[ds][m] for ds in dss])
        byc = np.array([benef_c[ds][m] for ds in dss])
        summary.append(dict(method=m,
                            pearson_raw=round(_pearson(ex, by), 3),
                            spearman_raw=round(_spearman(ex, by), 3),
                            pearson_clip=round(_pearson(ex, byc), 3),
                            spearman_clip=round(_spearman(ex, byc), 3)))

    # ---- console ----
    print(f"\n=== teacher edge vs benefit  ({os.path.basename(args.exp_dir)}, n={len(dss)} datasets) ===")
    print("teacher_edge = rmse_unguided - rmse_M20 (per-dataset mean):")
    for ds in dss:
        print(f"  {ds:11s} edge={edge[ds][0]:+.3f}  | " +
              " ".join(f"{m}:{benef[ds][m]:+.3f}" for m in methods))
    print(f"\ncorrelation across datasets (Spearman rho is robust; clip={args.clip}):")
    print(f"  {'method':18s} {'pearson':>8s} {'spearman':>9s} {'pears_clip':>11s} {'spear_clip':>11s}")
    for s in summary:
        print(f"  {s['method']:18s} {s['pearson_raw']:+8.3f} {s['spearman_raw']:+9.3f} "
              f"{s['pearson_clip']:+11.3f} {s['spearman_clip']:+11.3f}")

    # ---- csv ----
    with open(os.path.join(args.exp_dir, "teacher_edge_corr.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    # ---- scatter (one panel per method) ----
    ncol = min(3, len(methods)); nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow), squeeze=False)
    for i, m in enumerate(methods):
        ax = axes[i // ncol][i % ncol]
        by = np.array([benef[ds][m] for ds in dss])
        ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
        ax.scatter(ex, by, c="#0984e3", zorder=3)
        for ds, x, y in zip(dss, ex, by):
            ax.annotate(ds, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_title(f"{m}\nPearson r={summary[i]['pearson_raw']:+.2f} | Spearman={summary[i]['spearman_raw']:+.2f}",
                     fontsize=9)
        ax.set_xlabel("teacher edge (unguided - M20 RMSE)"); ax.set_ylabel(f"benefit (impr_{m})")
        ax.grid(alpha=0.25)
    for j in range(len(methods), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Teacher accuracy vs distillation benefit — {os.path.basename(args.exp_dir)}", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.exp_dir, "teacher_edge_corr.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- markdown ----
    with open(os.path.join(args.exp_dir, "teacher_edge_corr.md"), "w") as fh:
        fh.write(f"# Teacher edge vs distillation benefit ({os.path.basename(args.exp_dir)})\n\n")
        fh.write("`teacher_edge = rmse_unguided - rmse_M20` (>0 = teacher is the more accurate model). "
                 "`benefit = impr` (>0 = mode beats no-teacher). One point per dataset.\n\n")
        fh.write("| dataset | teacher_edge | " + " | ".join(methods) + " |\n")
        fh.write("|" + "---|" * (len(methods) + 2) + "\n")
        for ds in dss:
            fh.write(f"| {ds} | {edge[ds][0]:+.3f} | " +
                     " | ".join(f"{benef[ds][m]:+.3f}" for m in methods) + " |\n")
        fh.write("\n## Correlation across datasets\n\n")
        fh.write("| method | Pearson r | Spearman rho | Pearson (clipped) | Spearman (clipped) |\n|---|---|---|---|---|\n")
        for s in summary:
            fh.write(f"| {s['method']} | {s['pearson_raw']:+.3f} | {s['spearman_raw']:+.3f} | "
                     f"{s['pearson_clip']:+.3f} | {s['spearman_clip']:+.3f} |\n")
        fh.write(f"\nn = {len(dss)} datasets (correlation is illustrative, not significant at this n). "
                 "A strongly positive r would mean 'more accurate teacher -> more benefit'; "
                 "near-zero / negative refutes teacher accuracy as a predictor.\n")

    print(f"\nWrote teacher_edge_corr.{{png,md,csv}} to {args.exp_dir}")


if __name__ == "__main__":
    main()
