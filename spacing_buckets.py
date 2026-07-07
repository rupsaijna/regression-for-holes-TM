"""Experiment 12 / Tier B — per-region spacing analysis (the PROMPT-1 question).

For each dataset x seed, train the teacher (M20), the unguided 30%-hole student, and the best
guided students (force_ii, teacher_sign_tmag). Then, for every held-out TEST point, measure its
local SPACING = distance to the nearest SURVIVING (30%-hole) training point in standardized
feature space, and bucket test points by spacing decile. Per bucket, report mean |error| for each
model and the teacher BENEFIT (|err_unguided| - |err_guided|).

Tests:
  B1 (founding question): does per-point error rise with spacing? (per model; Spearman over points)
  B2 (mechanism): does distillation's benefit concentrate in HIGH-spacing regions, or not?
                  (Spearman of benefit vs spacing; bucket curve)
Also records hole_dist = distance to nearest REMOVED train point (proximity to a masked hole).

Out: outputs/distillation/exp12/{<ds>__spacing_buckets.csv, spacing_summary.{csv,md},
     error_vs_spacing__<ds>.png, benefit_vs_spacing__<ds>.png}
"""
import argparse, csv, logging, os, sys, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer

import config
from common import datasets, metrics
from common.data import get_split_data
from experiment5_distillation import nested_structured, teacher_encoded_predictions
from experiment11_tmag import train_student              # v4 harness (opposite_mode/same_side_tmag)

# models compared per point: name -> (opposite_mode, same_side_tmag) ; None teacher_vec => unguided
GUIDED = {"force_ii": ("force_ii", False), "teacher_sign_tmag": ("teacher_sign_tmag", False)}
NBUCKETS = 10


def run_dataset(key, args):
    Xtr, Xte, ytr, yte = get_split_data(key)
    binr = StandardBinarizer(max_bits_per_feature=config.TM_MAX_BITS_PER_FEATURE)
    Xb, Xb_te = binr.fit_transform(Xtr).astype(np.uint32), binr.transform(Xte).astype(np.uint32)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)         # spacing measured in z-scored space
    g_min, g_max, n = float(ytr.min()), float(ytr.max()), len(ytr)
    print(f"\n=== {key}: {n} train / {len(yte)} test ===", flush=True)

    pts = []     # per (seed, test-point) rows
    for seed in [config.RANDOM_SEED + i for i in range(args.seeds)]:
        rem_t, rem_s = nested_structured(Xtr_s, [0.2, 0.3], seed, hole_frac=0.05)
        kept_t, kept_s = np.setdiff1d(np.arange(n), rem_t), np.setdiff1d(np.arange(n), rem_s)
        Xb_s, y_s = Xb[kept_s], ytr[kept_s]
        M20 = train_student(Xb[kept_t], ytr[kept_t], args.epochs, g_min, g_max, seed)
        M30u = train_student(Xb_s, y_s, args.epochs, g_min, g_max, seed)
        tvec = teacher_encoded_predictions(M20, Xb_s)
        guided = {m: train_student(Xb_s, y_s, args.epochs, g_min, g_max, seed, tvec, args.f,
                                   args.p_teacher, opposite_mode=cfg[0], same_side_tmag=cfg[1])
                  for m, cfg in GUIDED.items()}

        spacing = metrics.spacing_nn_to_train(Xte_s, Xtr_s[kept_s], k=1)    # dist to surviving train
        hole_dist = metrics.spacing_nn_to_train(Xte_s, Xtr_s[rem_s], k=1)   # dist to a masked hole
        e_ung = np.abs(yte - M30u.predict(Xb_te))
        e_m20 = np.abs(yte - M20.predict(Xb_te))
        e_g = {m: np.abs(yte - guided[m].predict(Xb_te)) for m in GUIDED}
        for i in range(len(yte)):
            row = dict(seed=seed, spacing=float(spacing[i]), hole_dist=float(hole_dist[i]),
                       err_unguided=float(e_ung[i]), err_M20=float(e_m20[i]))
            for m in GUIDED:
                row[f"err_{m}"] = float(e_g[m][i])
                row[f"benefit_{m}"] = float(e_ung[i] - e_g[m][i])   # >0 = guided beats unguided here
            pts.append(row)
        print(f"  seed {seed}: spacing[min/med/max]="
              f"{spacing.min():.3f}/{np.median(spacing):.3f}/{spacing.max():.3f}", flush=True)
    return pts


def summarize(key, pts, out):
    sp = np.array([p["spacing"] for p in pts])
    hd = np.array([p["hole_dist"] for p in pts])
    models = ["unguided", "M20"] + list(GUIDED)
    # --- per-point Spearman (B1: error vs spacing ; B2: benefit vs spacing) ---
    srow = {"dataset": key, "n_points": len(pts)}
    for m in models:
        e = np.array([p[f"err_{m}"] for p in pts])
        rho, pval = stats.spearmanr(sp, e)
        srow[f"rho_err_{m}"] = round(float(rho), 3); srow[f"p_err_{m}"] = round(float(pval), 4)
    for m in GUIDED:
        b = np.array([p[f"benefit_{m}"] for p in pts])
        rho, pval = stats.spearmanr(sp, b)
        srow[f"rho_benefit_{m}"] = round(float(rho), 3); srow[f"p_benefit_{m}"] = round(float(pval), 4)

    # --- B3: in-hole vs out-of-hole split (a test point is "in a hole" if its nearest REMOVED
    #     train point is closer than its nearest SURVIVING train point -> its local data was carved) ---
    in_hole = hd < sp
    srow["frac_in_hole"] = round(float(in_hole.mean()), 3)
    for grp, sel in [("inhole", in_hole), ("outhole", ~in_hole)]:
        for m in models:
            e = np.array([p[f"err_{m}"] for p in pts])
            srow[f"err_{m}_{grp}"] = round(float(e[sel].mean()), 4) if sel.any() else ""
        for m in GUIDED:
            b = np.array([p[f"benefit_{m}"] for p in pts])
            srow[f"benefit_{m}_{grp}"] = round(float(b[sel].mean()), 4) if sel.any() else ""

    # --- persist per-point rows (enables future per-point analyses with no retraining) ---
    fields = ["seed", "spacing", "hole_dist", "in_hole",
              "err_unguided", "err_M20"] + [f"err_{m}" for m in GUIDED] + [f"benefit_{m}" for m in GUIDED]
    with open(os.path.join(out, f"{key}__points.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for p, ih in zip(pts, in_hole):
            w.writerow({k: (int(ih) if k == "in_hole" else round(p[k], 6)) for k in fields})

    # --- decile buckets (B1/B2 curves) ---
    edges = np.quantile(sp, np.linspace(0, 1, NBUCKETS + 1))
    edges[-1] += 1e-9
    bucket = np.clip(np.digitize(sp, edges) - 1, 0, NBUCKETS - 1)
    brows = []
    for b in range(NBUCKETS):
        sel = bucket == b
        if not sel.any():
            continue
        rec = dict(dataset=key, bucket=b, n=int(sel.sum()),
                   spacing_mean=round(float(sp[sel].mean()), 4),
                   hole_dist_mean=round(float(np.array([p["hole_dist"] for p in pts])[sel].mean()), 4))
        for m in models:
            rec[f"err_{m}"] = round(float(np.array([p[f"err_{m}"] for p in pts])[sel].mean()), 4)
        for m in GUIDED:
            rec[f"benefit_{m}"] = round(float(np.array([p[f"benefit_{m}"] for p in pts])[sel].mean()), 4)
        brows.append(rec)

    with open(os.path.join(out, f"{key}__spacing_buckets.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)
    _plots(key, brows, out)
    return srow


def _plots(key, brows, out):
    x = [r["spacing_mean"] for r in brows]
    # error vs spacing (B1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c in [("unguided", "#b2bec3"), ("M20", "#6c5ce7"),
                 ("force_ii", "#0984e3"), ("teacher_sign_tmag", "#00b894")]:
        ax.plot(x, [r[f"err_{m}"] for r in brows], "-o", color=c, label=m, ms=4)
    ax.set_xlabel("local spacing (dist to nearest surviving train point, z-space)")
    ax.set_ylabel("mean |error|"); ax.set_title(f"{key}: error vs spacing (B1)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"error_vs_spacing__{key}.png"), dpi=120); plt.close(fig)
    # benefit vs spacing (B2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c in [("force_ii", "#0984e3"), ("teacher_sign_tmag", "#00b894")]:
        ax.plot(x, [r[f"benefit_{m}"] for r in brows], "-o", color=c, label=m, ms=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("local spacing (dist to nearest surviving train point, z-space)")
    ax.set_ylabel("teacher benefit (|err_ung| - |err_guided|)")
    ax.set_title(f"{key}: distillation benefit vs spacing (B2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"benefit_vs_spacing__{key}.png"), dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ccpp", "airquality", "california", "energy"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--f", type=float, default=0.75)
    ap.add_argument("--p-teacher", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "distillation", "exp12"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = []
    for key in args.datasets:
        pts = run_dataset(key, args)
        summary.append(summarize(key, pts, args.out))

    with open(os.path.join(args.out, "spacing_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print("\n===== SUMMARY: Spearman(spacing, .) over test points =====")
    print("  B1 = error vs spacing (>0 => sparser = worse);  B2 = benefit vs spacing (>0 => helps more when sparse)")
    for s in summary:
        print(f"\n  {s['dataset']} (n={s['n_points']}, in-hole frac={s['frac_in_hole']}):")
        print("    B1 err :  " + "  ".join(
            f"{m}={s['rho_err_'+m]:+.2f}" for m in ['unguided', 'M20'] + list(GUIDED)))
        print("    B2 bnft:  " + "  ".join(f"{m}={s['rho_benefit_'+m]:+.2f}" for m in GUIDED))
        print("    B3 benefit in/out hole:  " + "  ".join(
            f"{m}={s['benefit_'+m+'_inhole']:+.3f}/{s['benefit_'+m+'_outhole']:+.3f}" for m in GUIDED))

    with open(os.path.join(args.out, "spacing_summary.md"), "w") as fh:
        fh.write("# E12 Tier B — error & distillation benefit vs local spacing\n\n")
        fh.write("Spearman over held-out test points (pooled across seeds). spacing = distance to "
                 "nearest surviving (30%-hole) train point, z-scored feature space.\n\n")
        fh.write("B1: does error rise with spacing? (rho_err > 0 = sparser regions learned worse)\n\n")
        fh.write("| dataset | n | err unguided | err M20 | err force_ii | err tsign_tmag |\n|---|---|---|---|---|---|\n")
        for s in summary:
            fh.write(f"| {s['dataset']} | {s['n_points']} | {s['rho_err_unguided']:+.2f} | "
                     f"{s['rho_err_M20']:+.2f} | {s['rho_err_force_ii']:+.2f} | "
                     f"{s['rho_err_teacher_sign_tmag']:+.2f} |\n")
        fh.write("\nB2: does benefit concentrate where it's sparse? (rho_benefit > 0 = helps more in sparse regions)\n\n")
        fh.write("| dataset | benefit force_ii | benefit tsign_tmag |\n|---|---|---|\n")
        for s in summary:
            fh.write(f"| {s['dataset']} | {s['rho_benefit_force_ii']:+.2f} | "
                     f"{s['rho_benefit_teacher_sign_tmag']:+.2f} |\n")
        fh.write("\nB3: benefit IN-hole vs OUT-of-hole (a test point is in-hole if its nearest REMOVED "
                 "train point is closer than its nearest surviving one). If distillation fills holes, "
                 "in-hole benefit > out-of-hole.\n\n")
        fh.write("| dataset | frac in-hole | force_ii in/out | tsign_tmag in/out |\n|---|---|---|---|\n")
        for s in summary:
            fh.write(f"| {s['dataset']} | {s['frac_in_hole']} | "
                     f"{s['benefit_force_ii_inhole']:+.3f} / {s['benefit_force_ii_outhole']:+.3f} | "
                     f"{s['benefit_teacher_sign_tmag_inhole']:+.3f} / {s['benefit_teacher_sign_tmag_outhole']:+.3f} |\n")
    print(f"\n-> {args.out}/spacing_summary.{{csv,md}} + per-dataset buckets & plots")


if __name__ == "__main__":
    main()
