"""Correlate teacher EFFECTIVENESS (experiment 5) against per-dataset DENSITY
metrics -- every effectiveness-metric x density-metric pairing, to find what (if
anything) predicts when the teacher helps.

Reads:
  outputs/distillation/sweep__<ds>__structured.csv   (focused config rows)
  outputs/distillation/density_metrics.csv
Writes:
  outputs/distillation/effectiveness_metrics.csv
  outputs/distillation/eff_density_correlations.csv   (every pair, spearman+pearson)
  outputs/distillation/eff_density_heatmap.png
  outputs/distillation/eff_density_top_scatters.png
  outputs/distillation/eff_density_summary.md

Focused config: epochs=100, student_frac=0.3, f=0.75, f_opposite=0, p_teacher=0.5.
NOTE (future work): only one config / one student mask / structured only so far.
"""
import csv
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config

DDIR = os.path.join(config.OUTPUTS_DIR, "distillation")
EPOCHS, SF, F = 100, 0.3, 0.75


def num(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


def effectiveness_rows():
    """Per-dataset effectiveness metrics, aggregated over seeds, from the focused config."""
    rows = []
    for path in sorted(glob.glob(os.path.join(DDIR, "sweep__*__structured.csv"))):
        recs = [r for r in csv.DictReader(open(path))
                if int(r["epochs"]) == EPOCHS and abs(num(r["student_frac"]) - SF) < 1e-9
                and abs(num(r["f"]) - F) < 1e-9]
        if not recs:
            continue
        ds = recs[0]["dataset"]
        u = np.array([num(r["rmse_unguided"]) for r in recs])
        g = np.array([num(r["rmse_guided"]) for r in recs])
        m100 = np.array([num(r["rmse_M100"]) for r in recs])
        m20 = np.array([num(r["rmse_M20"]) for r in recs])
        d = u - g                                              # >0 == teacher helps
        with np.errstate(divide="ignore", invalid="ignore"):
            gapc = np.where((u - m100) != 0, d / (u - m100), np.nan)
            vteach = np.where((u - m20) != 0, d / (u - m20), np.nan)
        rows.append(dict(
            dataset=ds, n_seeds=len(recs),
            eff_raw=round(float(np.mean(d)), 5),                      # mean RMSE improvement
            eff_raw_median=round(float(np.median(d)), 5),
            eff_norm=round(float(np.mean(d / u)), 5),                 # fractional improvement
            eff_winrate=round(float(np.mean(g < u)), 5),             # frac seeds teacher helps
            eff_gapclosed=round(float(np.nanmean(gapc)), 5),          # toward M100 (noisy denom)
            eff_vs_teacher=round(float(np.nanmean(vteach)), 5),       # of the teacher's headroom
            eff_beat_m100=round(float(np.mean(m100 - g)), 5),        # guided vs full-data model
            rmse_unguided=round(float(np.mean(u)), 5),
            rmse_guided=round(float(np.mean(g)), 5)))
    return rows


def load_density():
    path = os.path.join(DDIR, "density_metrics.csv")
    return {r["dataset"]: r for r in csv.DictReader(open(path))}


EFF_KEYS = ["eff_raw", "eff_raw_median", "eff_norm", "eff_winrate",
            "eff_gapclosed", "eff_vs_teacher", "eff_beat_m100"]
DENS_KEYS = ["n", "d", "spread", "nn1_mean", "knn5_mean", "knn5_median",
             "het_global_cv", "knn_cv_mean", "knn_ratio_mean", "norm_entropy",
             "min_pairwise", "spc_err_strength", "spc_err_nn_k1", "var_err_strength"]


def main():
    eff = effectiveness_rows()
    if len(eff) < 3:
        print(f"Only {len(eff)} datasets have focused-config results so far; need more. "
              f"(re-run when sweeps finish)")
    dens = load_density()
    # join
    data = []
    for e in eff:
        d = dens.get(e["dataset"])
        if d:
            rec = dict(e)
            rec.update({k: num(d.get(k, "")) for k in DENS_KEYS})
            data.append(rec)
    keys = [r["dataset"] for r in data]
    print(f"Datasets joined ({len(data)}): {', '.join(keys)}")

    # write effectiveness table
    with open(os.path.join(DDIR, "effectiveness_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(eff[0].keys())); w.writeheader(); w.writerows(eff)

    if len(data) < 4:
        print("Too few joined datasets for correlation; effectiveness table written, stopping.")
        return

    # every (eff x dens) correlation, with and without tiny-n datasets
    def corr_table(rows):
        out = []
        for ek in EFF_KEYS:
            for dk in DENS_KEYS:
                x = np.array([r[ek] for r in rows], float)
                y = np.array([r[dk] for r in rows], float)
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 4 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
                    continue
                sr, sp = stats.spearmanr(x[m], y[m])
                pr, pp = stats.pearsonr(x[m], y[m])
                out.append(dict(eff=ek, density=dk, n=int(m.sum()),
                                spearman=round(float(sr), 3), spearman_p=round(float(sp), 3),
                                pearson=round(float(pr), 3), pearson_p=round(float(pp), 3)))
        return out

    big = [r for r in data if r["n"] >= 100]
    full = corr_table(data)
    nobig = corr_table(big)
    with open(os.path.join(DDIR, "eff_density_correlations.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subset", "eff", "density", "n", "spearman",
                                           "spearman_p", "pearson", "pearson_p"])
        w.writeheader()
        for r in full: w.writerow(dict(r, subset="all"))
        for r in nobig: w.writerow(dict(r, subset="n>=100"))

    # heatmap of spearman (all datasets)
    M = np.full((len(EFF_KEYS), len(DENS_KEYS)), np.nan)
    lut = {(r["eff"], r["density"]): r["spearman"] for r in full}
    for i, ek in enumerate(EFF_KEYS):
        for j, dk in enumerate(DENS_KEYS):
            if (ek, dk) in lut: M[i, j] = lut[(ek, dk)]
    fig, ax = plt.subplots(figsize=(0.7 * len(DENS_KEYS) + 3, 0.6 * len(EFF_KEYS) + 2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DENS_KEYS))); ax.set_xticklabels(DENS_KEYS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(EFF_KEYS))); ax.set_yticklabels(EFF_KEYS, fontsize=8)
    for i in range(len(EFF_KEYS)):
        for j in range(len(DENS_KEYS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
    fig.colorbar(im, label="Spearman r")
    ax.set_title(f"Teacher effectiveness vs density — Spearman ({len(data)} datasets)")
    fig.tight_layout(); fig.savefig(os.path.join(DDIR, "eff_density_heatmap.png"), dpi=120); plt.close(fig)

    # rank strongest pairs
    ranked = sorted(full, key=lambda r: -abs(r["spearman"]))
    print("\nStrongest |Spearman| pairs (all datasets):")
    for r in ranked[:10]:
        print(f"  {r['eff']:14s} vs {r['density']:16s}  rho={r['spearman']:+.2f} (p={r['spearman_p']}) "
              f"pearson={r['pearson']:+.2f}")

    # scatter the top 4 pairs
    top = ranked[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, r in zip(axes.ravel(), top):
        x = np.array([row[r["density"]] for row in data], float)
        y = np.array([row[r["eff"]] for row in data], float)
        ax.scatter(x, y)
        for row in data:
            ax.annotate(row["dataset"], (row[r["density"]], row[r["eff"]]), fontsize=7)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel(r["density"]); ax.set_ylabel(r["eff"])
        ax.set_title(f"{r['eff']} vs {r['density']}  (rho={r['spearman']:+.2f})")
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(DDIR, "eff_density_top_scatters.png"), dpi=120); plt.close(fig)

    # markdown summary
    with open(os.path.join(DDIR, "eff_density_summary.md"), "w") as fh:
        fh.write(f"# Teacher effectiveness vs data density ({len(data)} datasets, focused config)\n\n")
        fh.write(f"Config: epochs={EPOCHS}, student_frac={SF}, f={F}, f_opposite=0, p_teacher=0.5, "
                 f"structured masking. eff_raw>0 = teacher lowers RMSE.\n\n")
        fh.write("## Strongest |Spearman| pairings (all datasets)\n\n")
        fh.write("| effectiveness | density | Spearman | p | Pearson |\n|---|---|---|---|---|\n")
        for r in ranked[:12]:
            fh.write(f"| {r['eff']} | {r['density']} | {r['spearman']:+.2f} | {r['spearman_p']} | {r['pearson']:+.2f} |\n")
        fh.write("\n_NOTE: single config / one student mask / structured only; tiny-n datasets "
                 "(bloodfat, mortality) included — see n>=100 subset in eff_density_correlations.csv. "
                 "Future work: more configs, student masks, uniform mask, more seeds._\n")
    print(f"\n-> {DDIR}/eff_density_*.{{csv,png,md}}")


if __name__ == "__main__":
    main()
