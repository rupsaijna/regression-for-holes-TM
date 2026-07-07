"""Experiment 12 / Tier A — does data spacing/density predict distillation benefit across datasets,
for the E8-E11 modes? And does it predict better than teacher accuracy (the teacher edge)?

Reads:
  <exp-dir>/<ds>__tmag.csv   (broad experiment11_tmag run: impr_<mode>, rmse_unguided/M20/M00)
  outputs/distillation/density_metrics.csv   (per-dataset spacing/density battery)
Computes per dataset x mode effectiveness (eff_raw, eff_norm, eff_winrate, beat_m00), plus the
per-dataset teacher_edge and a less_is_more flag, joins the density battery, and correlates each
effectiveness metric against every PREDICTOR (density metrics + teacher_edge) across datasets.

Out: <exp-dir>/modes_effectiveness.csv, modes_density_correlations.csv, modes_density_summary.md,
     modes_density_heatmap__<eff>.png
Windows-runnable (no tmu).
"""
import argparse, csv, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config

DENS_KEYS = ["n", "d", "spread", "nn1_mean", "knn5_mean", "knn5_median", "het_global_cv",
             "knn_cv_mean", "knn_ratio_mean", "norm_entropy", "min_pairwise",
             "spc_err_strength", "spc_err_nn_k1", "var_err_strength"]
EFF_KEYS = ["eff_raw", "eff_norm", "eff_winrate", "beat_m00"]


def num(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


def load_density(ddir):
    path = os.path.join(ddir, "density_metrics.csv")
    return {r["dataset"]: r for r in csv.DictReader(open(path))} if os.path.exists(path) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="outputs/distillation/exp12_tierA")
    ap.add_argument("--density", default=os.path.join(config.OUTPUTS_DIR, "distillation"))
    ap.add_argument("--min-n", type=int, default=0,
                    help="drop datasets with train n < this from the CORRELATION (CSV keeps all)")
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(os.path.join(args.exp_dir, "*__tmag.csv")))]
    if not paths:
        sys.exit(f"No <ds>__tmag.csv in {args.exp_dir} (run the broad experiment11_tmag sweep there first)")
    dens = load_density(args.density)

    modes = None
    eff_rows = []          # per dataset x mode
    ds_meta = {}           # dataset -> teacher_edge, less_is_more
    for path in paths:
        rows = list(csv.DictReader(open(path, newline="")))
        if not rows:
            continue
        ds = rows[0]["dataset"]
        if modes is None:
            modes = [k[len("impr_"):] for k in rows[0] if k.startswith("impr_")]
        u = np.array([num(r["rmse_unguided"]) for r in rows])
        edge = np.nan
        if "rmse_M20" in rows[0]:
            edge = float(np.mean(u - np.array([num(r["rmse_M20"]) for r in rows])))
        lim = np.nan
        m00 = np.array([num(r.get("rmse_M00", "nan")) for r in rows])
        if np.isfinite(m00).all():
            lim = float((np.mean(m00) - np.mean(u)) / np.mean(u))   # normalized; >0 => full WORSE than unguided => less-is-more
        ds_meta[ds] = dict(teacher_edge=round(edge, 5) if edge == edge else "",
                           less_is_more_margin=round(lim, 5) if lim == lim else "")
        for m in modes:
            impr = np.array([num(r[f"impr_{m}"]) for r in rows])
            mode_rmse = np.array([num(r[f"rmse_{m}"]) for r in rows])
            eff_rows.append(dict(
                dataset=ds, mode=m, n_seeds=len(rows),
                eff_raw=round(float(np.mean(impr)), 5),
                eff_norm=round(float(np.mean(impr / u)), 5),
                eff_winrate=round(float(np.mean(impr > 0)), 5),
                beat_m00=round(float(np.mean(m00 - mode_rmse)), 5) if np.isfinite(m00).all() else ""))

    # join density + teacher_edge predictors
    PRED_KEYS = DENS_KEYS + ["teacher_edge"]
    for e in eff_rows:
        d = dens.get(e["dataset"], {})
        for k in DENS_KEYS:
            e[k] = num(d.get(k, ""))
        e["teacher_edge"] = num(ds_meta[e["dataset"]]["teacher_edge"])
        e["less_is_more_margin"] = ds_meta[e["dataset"]]["less_is_more_margin"]

    with open(os.path.join(args.exp_dir, "modes_effectiveness.csv"), "w", newline="") as fh:
        cols = ["dataset", "mode", "n_seeds"] + EFF_KEYS + ["less_is_more_margin"] + PRED_KEYS
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(eff_rows)

    # correlation subset: optionally drop tiny-n datasets (effectiveness CSV above keeps all)
    crows = [e for e in eff_rows if (not args.min_n) or
             (np.isfinite(num(e.get("n", ""))) and num(e["n"]) >= args.min_n)]
    datasets_seen = sorted({e["dataset"] for e in crows})
    print(f"Datasets in correlation ({len(datasets_seen)}, min_n={args.min_n}): {', '.join(datasets_seen)}")
    print(f"Modes: {', '.join(modes)}")
    if len(datasets_seen) < 4:
        print("Too few datasets for correlation; effectiveness table written, stopping.")
        return

    # correlations: per mode, each eff metric vs each predictor, across datasets
    corr = []
    for m in modes:
        sub = [e for e in crows if e["mode"] == m]
        for ek in EFF_KEYS:
            y = np.array([num(e[ek]) for e in sub], float)
            for pk in PRED_KEYS:
                x = np.array([num(e[pk]) for e in sub], float)
                msk = np.isfinite(x) & np.isfinite(y)
                if msk.sum() < 4 or np.std(x[msk]) == 0 or np.std(y[msk]) == 0:
                    continue
                sr, sp = stats.spearmanr(x[msk], y[msk])
                pr, pp = stats.pearsonr(x[msk], y[msk])
                corr.append(dict(mode=m, eff=ek, predictor=pk, n=int(msk.sum()),
                                 spearman=round(float(sr), 3), spearman_p=round(float(sp), 3),
                                 pearson=round(float(pr), 3)))
    with open(os.path.join(args.exp_dir, "modes_density_correlations.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mode", "eff", "predictor", "n", "spearman", "spearman_p", "pearson"])
        w.writeheader(); w.writerows(corr)

    # heatmap of Spearman(eff_norm, predictor) : modes x predictors
    _heatmap(modes, PRED_KEYS, corr, "eff_norm", os.path.join(args.exp_dir, "modes_density_heatmap__eff_norm.png"))

    # rank strongest, and compare best density predictor vs teacher_edge for eff_norm
    ranked = sorted([c for c in corr if c["eff"] == "eff_norm"], key=lambda r: -abs(r["spearman"]))
    print("\nStrongest |Spearman| (eff_norm vs predictor), any mode:")
    for r in ranked[:10]:
        print(f"  {r['mode']:16s} vs {r['predictor']:16s} rho={r['spearman']:+.2f} (p={r['spearman_p']})")

    with open(os.path.join(args.exp_dir, "modes_density_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# E12 Tier A - spacing/density vs distillation benefit ({len(datasets_seen)} datasets)\n\n")
        fh.write(f"Datasets: {', '.join(datasets_seen)}. Modes: {', '.join(modes)}.\n\n")
        fh.write("## Strongest |Spearman| pairings for eff_norm (fractional benefit)\n\n")
        fh.write("| mode | predictor | Spearman | p | Pearson |\n|---|---|---|---|---|\n")
        for r in ranked[:15]:
            fh.write(f"| {r['mode']} | {r['predictor']} | {r['spearman']:+.2f} | {r['spearman_p']} | {r['pearson']:+.2f} |\n")
        # density vs teacher_edge head-to-head per mode
        fh.write("\n## Best DENSITY predictor vs teacher_edge (eff_norm), per mode\n\n")
        fh.write("| mode | best density predictor | rho | teacher_edge rho |\n|---|---|---|---|\n")
        for m in modes:
            cm = [c for c in corr if c["mode"] == m and c["eff"] == "eff_norm"]
            dens_c = [c for c in cm if c["predictor"] != "teacher_edge"]
            te = next((c["spearman"] for c in cm if c["predictor"] == "teacher_edge"), float("nan"))
            if dens_c:
                best = max(dens_c, key=lambda c: abs(c["spearman"]))
                fh.write(f"| {m} | {best['predictor']} | {best['spearman']:+.2f} | {te:+.2f} |\n")
        fh.write("\n_NOTE: small n datasets; correlations illustrative. less_is_more_margin "
                 "(rmse_M00 - rmse_unguided > 0) is in modes_effectiveness.csv per dataset._\n")
    print(f"\n-> {args.exp_dir}/modes_*.{{csv,md,png}}")


def _heatmap(modes, preds, corr, eff, path):
    lut = {(c["mode"], c["predictor"]): c["spearman"] for c in corr if c["eff"] == eff}
    M = np.full((len(modes), len(preds)), np.nan)
    for i, m in enumerate(modes):
        for j, p in enumerate(preds):
            if (m, p) in lut: M[i, j] = lut[(m, p)]
    fig, ax = plt.subplots(figsize=(0.6 * len(preds) + 3, 0.6 * len(modes) + 2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(preds))); ax.set_xticklabels(preds, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(modes))); ax.set_yticklabels(modes, fontsize=8)
    for i in range(len(modes)):
        for j in range(len(preds)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
    fig.colorbar(im, label="Spearman r")
    ax.set_title(f"Spearman({eff}, predictor) — teacher_edge is the rightmost column")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
