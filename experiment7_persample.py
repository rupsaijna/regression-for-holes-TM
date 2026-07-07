"""Experiment 7 — PER-SAMPLE analysis: which TEST points does the teacher rescue?

Dataset-level correlation of teacher effectiveness was a null (n=12, noisy). Here we
go to per-test-point resolution: for the focused config (teacher 20% mask -> student
30% mask, structured, f=0.75, f_opposite=0, p=0.5, 100ep), train M20 (teacher),
M30u (unguided student), M30g (guided student); for every test point e and seed:

  teacher_help[e] = |y - pred_M30u| - |y - pred_M30g|     (>0 = teacher cut this point's error)

and regress it on per-point LOCAL descriptors:
  in_hole          : Phase-7 region split (>= half of e's 5 nearest ORIG-train nbrs removed by student mask)
  local_density    : mean dist of e to its 5 nearest ORIG-train points (low = dense support)   [seed-indep]
  dist_nearest     : dist to nearest ORIG-train point                                            [seed-indep]
  local_y_rough    : std(y over e's 5 nearest train nbrs)/std(y)  (label roughness)              [seed-indep]
  support_loss     : dist(nearest student-kept) - dist(nearest orig)  (support the student lost) [per-seed]
  teacher_extra    : dist(nearest student-kept) - dist(nearest teacher-kept) (>0: teacher has closer support)
  ts_disagree      : |pred_M20 - pred_M30u| / std(y)   (teacher<->student disagreement at e)     [per-seed]
  err_unguided     : |y - pred_M30u| / std(y)          (how hard e already is)                   [per-seed]

n = seeds x n_test points per dataset (thousands), so the correlations have real power.

Outputs: outputs/distillation/persample/{<ds>__persample.csv, persample_correlations.csv,
         persample_inhole.csv, persample__<ds>.png, persample_summary.md}
"""
import argparse
import csv
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer

import config
from common import datasets, masking, metrics
from common.data import get_split_data, standardize
from experiment5_distillation import nested_structured, train_tm

DESCRIPTORS = ["in_hole", "local_density", "dist_nearest", "local_y_rough",
               "support_loss", "teacher_extra", "ts_disagree", "err_unguided"]


def _nearest(Xq, Xref, k=1):
    d, _ = NearestNeighbors(n_neighbors=k).fit(Xref).kneighbors(Xq)
    return d.mean(axis=1)


def run_dataset(key, args):
    Xtr_raw, Xte_raw, ytr, yte = get_split_data(key)
    binr = StandardBinarizer(max_bits_per_feature=config.TM_MAX_BITS_PER_FEATURE)
    Xb, Xb_te = binr.fit_transform(Xtr_raw).astype(np.uint32), binr.transform(Xte_raw).astype(np.uint32)
    _, Xs_tr, Xs_te = standardize(Xtr_raw, Xte_raw)
    g_min, g_max, ysd = float(ytr.min()), float(ytr.max()), float(np.std(ytr)) or 1.0
    n = len(ytr)
    print(f"\n=== {key}: {n} train / {len(yte)} test ===", flush=True)

    # seed-independent per-test-point descriptors
    local_density = metrics.spacing_nn_to_train(Xs_te, Xs_tr, k=5)
    dist_nearest = _nearest(Xs_te, Xs_tr, k=1)
    nn5 = NearestNeighbors(n_neighbors=5).fit(Xs_tr).kneighbors(Xs_te, return_distance=False)
    local_y_rough = ytr[nn5].std(axis=1) / ysd
    d_orig = dist_nearest

    rows = []
    for seed in [config.RANDOM_SEED + i for i in range(args.seeds)]:
        removed_t, removed_s = nested_structured(Xs_tr, [args.teacher_frac, args.student_frac],
                                                 seed, hole_frac=args.hole_frac)
        kept_t = np.setdiff1d(np.arange(n), removed_t)
        kept_s = np.setdiff1d(np.arange(n), removed_s)

        M20 = train_tm(Xb[kept_t], ytr[kept_t], args.epochs, g_min, g_max, seed)
        M30u = train_tm(Xb[kept_s], ytr[kept_s], args.epochs, g_min, g_max, seed)
        M30g = train_tm(Xb[kept_s], ytr[kept_s], args.epochs, g_min, g_max, seed,
                        teacher=M20, f=args.f, f_opposite=0.0, p_start=args.p_teacher)

        p20, p30u, p30g = M20.predict(Xb_te), M30u.predict(Xb_te), M30g.predict(Xb_te)
        err_u, err_g = np.abs(yte - p30u), np.abs(yte - p30g)
        in_hole = masking.region_split(Xs_tr, removed_s, Xs_te, k=5).astype(int)
        d_student = _nearest(Xs_te, Xs_tr[kept_s], 1)
        d_teacher = _nearest(Xs_te, Xs_tr[kept_t], 1)

        feat = dict(
            in_hole=in_hole,
            local_density=local_density, dist_nearest=dist_nearest, local_y_rough=local_y_rough,
            support_loss=d_student - d_orig, teacher_extra=d_student - d_teacher,
            ts_disagree=np.abs(p20 - p30u) / ysd, err_unguided=err_u / ysd)
        help_norm = (err_u - err_g) / ysd
        for i in range(len(yte)):
            row = dict(dataset=key, seed=seed, idx=i, teacher_help=round(float(help_norm[i]), 6))
            for dname in DESCRIPTORS:
                row[dname] = round(float(feat[dname][i]), 6)
            rows.append(row)
        print(f"  seed {seed}: mean help={help_norm.mean():+.4f}  "
              f"in-hole help={help_norm[in_hole == 1].mean():+.4f} ({(in_hole==1).sum()}) "
              f"surviving={help_norm[in_hole == 0].mean():+.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ccpp", "california", "energy", "airquality"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--f", type=float, default=0.75)
    ap.add_argument("--p-teacher", type=float, default=0.5)
    ap.add_argument("--teacher-frac", type=float, default=0.2)
    ap.add_argument("--student-frac", type=float, default=0.3)
    ap.add_argument("--hole-frac", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "distillation", "persample"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    corr_rows, inhole_rows = [], []
    for key in args.datasets:
        rows = run_dataset(key, args)
        with open(os.path.join(args.out, f"{key}__persample.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        help_v = np.array([r["teacher_help"] for r in rows])
        ih = np.array([r["in_hole"] for r in rows])
        inhole_rows.append(dict(dataset=key, n_points=len(rows),
                                help_in_hole=round(float(help_v[ih == 1].mean()) if (ih == 1).any() else float("nan"), 5),
                                help_surviving=round(float(help_v[ih == 0].mean()), 5),
                                frac_in_hole=round(float((ih == 1).mean()), 4)))
        for dname in DESCRIPTORS:
            x = np.array([r[dname] for r in rows])
            if np.std(x) == 0:
                continue
            rho, p = stats.spearmanr(x, help_v)
            corr_rows.append(dict(dataset=key, descriptor=dname, n=len(rows),
                                  spearman=round(float(rho), 3), p=round(float(p), 4)))
        _plot(key, rows, args.out)

    with open(os.path.join(args.out, "persample_correlations.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "descriptor", "n", "spearman", "p"])
        w.writeheader(); w.writerows(corr_rows)
    with open(os.path.join(args.out, "persample_inhole.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inhole_rows[0].keys())); w.writeheader(); w.writerows(inhole_rows)

    # consensus: mean Spearman per descriptor across datasets + sign consistency
    print("\n===== in-hole vs surviving teacher help (normalized) =====")
    for r in inhole_rows:
        print(f"  {r['dataset']:11s} in-hole={r['help_in_hole']:+.4f}  surviving={r['help_surviving']:+.4f}  "
              f"(in-hole frac {r['frac_in_hole']})")
    print("\n===== descriptor <-> teacher_help (Spearman), per dataset + consensus =====")
    print(f"  {'descriptor':14s} " + " ".join(f"{d:>10s}" for d in args.datasets) + "   mean  sign-consistent")
    for dname in DESCRIPTORS:
        vals = {r["dataset"]: r["spearman"] for r in corr_rows if r["descriptor"] == dname}
        rhos = [vals.get(d, np.nan) for d in args.datasets]
        finite = [r for r in rhos if r == r]
        mean = np.mean(finite) if finite else np.nan
        consistent = "yes" if finite and all(np.sign(r) == np.sign(finite[0]) for r in finite) else "no"
        print(f"  {dname:14s} " + " ".join(f"{(r if r==r else 0):+10.2f}" for r in rhos)
              + f"   {mean:+.2f}   {consistent}")
    _write_md(inhole_rows, corr_rows, args)
    print(f"\n-> {args.out}/")


def _plot(key, rows, out):
    help_v = np.array([r["teacher_help"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ih = np.array([r["in_hole"] for r in rows])
    axes[0].bar(["surviving", "in-hole"],
                [help_v[ih == 0].mean(), help_v[ih == 1].mean() if (ih == 1).any() else 0],
                color=["#b2bec3", "#0984e3"])
    axes[0].axhline(0, color="k", lw=0.8); axes[0].set_title(f"{key}: help by region"); axes[0].set_ylabel("mean teacher_help (norm)")
    for ax, dname in zip(axes[1:], ["teacher_extra", "ts_disagree"]):
        x = np.array([r[dname] for r in rows])
        q = np.quantile(x, np.linspace(0, 1, 11))
        binc = np.clip(np.digitize(x, q[1:-1]), 0, 9)
        m = [help_v[binc == b].mean() for b in range(10)]
        ax.plot(range(10), m, marker="o"); ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"{dname} decile"); ax.set_title(f"{key}: help vs {dname}"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"persample__{key}.png"), dpi=120); plt.close(fig)


def _write_md(inhole_rows, corr_rows, args):
    with open(os.path.join(args.out, "persample_summary.md"), "w") as fh:
        fh.write(f"# Experiment 7 — per-sample: which test points does the teacher rescue?\n\n")
        fh.write(f"Config: teacher {int(args.teacher_frac*100)}% -> student {int(args.student_frac*100)}% mask, "
                 f"structured, f={args.f}, p={args.p_teacher}, {args.epochs}ep, {args.seeds} seeds. "
                 f"teacher_help = (|err_unguided| - |err_guided|)/std(y) per test point; >0 = teacher helped.\n\n")
        fh.write("## In-hole vs surviving\n\n| dataset | in-hole | surviving | in-hole frac |\n|---|---|---|---|\n")
        for r in inhole_rows:
            fh.write(f"| {r['dataset']} | {r['help_in_hole']:+.4f} | {r['help_surviving']:+.4f} | {r['frac_in_hole']} |\n")
        fh.write("\n## Spearman(descriptor, teacher_help) per dataset\n\n| descriptor | "
                 + " | ".join(args.datasets) + " |\n|" + "---|" * (len(args.datasets) + 1) + "\n")
        for dname in DESCRIPTORS:
            vals = {r["dataset"]: r["spearman"] for r in corr_rows if r["descriptor"] == dname}
            fh.write(f"| {dname} | " + " | ".join(f"{vals.get(d, float('nan')):+.2f}" for d in args.datasets) + " |\n")


if __name__ == "__main__":
    main()
