"""Regenerate the two paper figures from existing CSVs (no retraining).
Fig 1: Tier-A Spearman heatmap (drops the blank knn_ratio_mean column).
Fig 2: per-region error-vs-spacing and benefit-vs-spacing on ccpp, both panels
       showing the same three versions (M30 + two guided modes) for consistency.
Outputs overwrite paper/figures/*.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------- Fig 1: Tier-A heatmap ----------------
corr = pd.read_csv(os.path.join(ROOT, "outputs", "distillation", "exp12_tierA",
                                 "modes_density_correlations.csv"))
dn = corr[corr.eff == "eff_norm"]
mode_order = ["uniform", "force_ii", "force_ii_tmag", "teacher_sign_tmag", "sameside_tmag"]
pred_order = ["n", "d", "spread", "nn1_mean", "knn5_mean", "knn5_median",
              "het_global_cv", "knn_cv_mean", "norm_entropy", "min_pairwise",
              "spc_err_strength", "spc_err_nn_k1", "var_err_strength", "teacher_edge"]
M = dn.pivot(index="mode", columns="predictor", values="spearman").reindex(
    index=mode_order, columns=pred_order)

fig, ax = plt.subplots(figsize=(13, 4.2))
im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(pred_order)))
ax.set_xticklabels(pred_order, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(mode_order)))
ax.set_yticklabels(mode_order, fontsize=10)
for i in range(len(mode_order)):
    for j in range(len(pred_order)):
        v = M.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "black")
ax.set_title("Spearman(benefit, predictor) across the 12-dataset panel "
             "(teacher_edge is the rightmost column)", fontsize=11)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("Spearman r")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "density_benefit_heatmap.png"), dpi=150)
plt.close(fig)
print("wrote heatmap; columns =", len(pred_order), "(no knn_ratio_mean)")

# ---------------- Fig 2: per-region spacing panels (ccpp) ----------------
pts = pd.read_csv(os.path.join(ROOT, "outputs", "distillation", "exp12",
                               "ccpp__points.csv"))
NB = 10
pts = pts.copy()
pts["bin"] = pd.qcut(pts["spacing"], NB, labels=False, duplicates="drop")
g = pts.groupby("bin")
x = g["spacing"].mean().values

# (a) error vs spacing: M30 (unguided) + two guided modes
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(x, g["err_unguided"].mean().values, "o-", color="#8c8c8c", label="M30")
ax.plot(x, g["err_force_ii"].mean().values, "o-", color="#1f77b4", label="force_ii")
ax.plot(x, g["err_teacher_sign_tmag"].mean().values, "o-", color="#2ca02c",
        label="teacher_sign_tmag")
ax.set_xlabel("local spacing (dist. to nearest surviving train point, z-space)")
ax.set_ylabel("mean |error|")
ax.set_title("ccpp: error vs spacing")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "error_vs_spacing__ccpp.png"), dpi=150)
plt.close(fig)

# (b) benefit vs spacing: same two guided modes + M30 zero reference line
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.axhline(0.0, color="#8c8c8c", lw=1.3, label="M30")
ax.plot(x, g["benefit_force_ii"].mean().values, "o-", color="#1f77b4", label="force_ii")
ax.plot(x, g["benefit_teacher_sign_tmag"].mean().values, "o-", color="#2ca02c",
        label="teacher_sign_tmag")
ax.set_xlabel("local spacing (dist. to nearest surviving train point, z-space)")
ax.set_ylabel("benefit (|err_M30| - |err_guided|)")
ax.set_title("ccpp: distillation benefit vs spacing")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "benefit_vs_spacing__ccpp.png"), dpi=150)
plt.close(fig)
print("wrote error/benefit panels; both show M30 + force_ii + teacher_sign_tmag")
