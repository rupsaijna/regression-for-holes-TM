"""Phase-7 masking VISUALIZATION: show the difference between

  * original training data vs masked training data (uniform vs structured holes)
  * original model result vs masked model result (spatial error + predicted-vs-actual)

at a chosen removal fraction `p`, for one model. Everything is drawn in a shared
2D PCA projection of the standardized feature space (fit on the FULL training set)
so the carved holes and the test-error geography line up panel-to-panel.

Produces a single 3x3 figure:
  Row 1  DATA      : original train | uniform-masked | structured-masked
                     (removed points drawn faint grey; kept points solid)
  Row 2  RESULT-map: baseline | uniform | structured -- test points coloured by
                     |y - y_hat| on a SHARED scale; in-hole test points ringed
  Row 3  RESULT-fit: predicted vs actual, with y=x and RMSE, for the same three

This reuses the exact masking utilities and model wrappers from the experiment,
so the picture is the experiment, not an approximation. It writes to a separate
file and never touches outputs/masking/*, so it is safe to run alongside a live
masking run.

Usage:
  python viz_masking.py --dataset energy --p 0.4
  python viz_masking.py --dataset ccpp --p 0.4 --model NN
  python viz_masking.py --dataset california --p 0.5 --model TM
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

import config
from common import datasets, masking, metrics
from common.data import get_split_data, standardize
from models.tm_model import TMModel
from models.nn_model import NNModel


def _make_model(name, seed):
    return TMModel() if name == "TM" else NNModel(seed=seed)


def _subsample(n, cap, seed=0):
    """Return an index array of <= cap points (for readable/fast scatter)."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="energy", choices=list(datasets.DATASETS))
    ap.add_argument("--p", type=float, default=0.4, help="removal fraction to visualize")
    ap.add_argument("--model", default="NN", choices=["TM", "NN"])
    ap.add_argument("--hole-frac", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="mask seed (matches experiment repeat r)")
    ap.add_argument("--max-points", type=int, default=4000, help="scatter subsample cap")
    args = ap.parse_args()

    key, p, model_name = args.dataset, args.p, args.model
    print(f"viz: dataset={key} p={p} model={model_name}", flush=True)

    Xtr, Xte, ytr, yte = get_split_data(key)
    scaler, Xtr_s, Xte_s = standardize(Xtr, Xte)
    n_train = len(ytr)

    # --- 2D projection shared by every panel (fit on full standardized train) ---
    if Xtr_s.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=config.RANDOM_SEED).fit(Xtr_s)
        Ptr, Pte = pca.transform(Xtr_s), pca.transform(Xte_s)
        ev = pca.explained_variance_ratio_
        ax_lbl = (f"PC1 ({ev[0]*100:.0f}%)", f"PC2 ({ev[1]*100:.0f}%)")
    else:  # single-feature fallback: feature vs target
        Ptr = np.column_stack([Xtr_s[:, 0], (ytr - ytr.mean()) / (ytr.std() or 1)])
        Pte = np.column_stack([Xte_s[:, 0], (yte - yte.mean()) / (yte.std() or 1)])
        ax_lbl = ("feature (z)", "target (z)")

    # --- masks (identical to the experiment) ---
    kept_u, removed_u = masking.mask_uniform(n_train, p, seed=args.seed)
    kept_s, removed_s = masking.mask_structured(Xtr_s, p, seed=args.seed, hole_frac=args.hole_frac)
    inhole_u = masking.region_split(Xtr_s, removed_u, Xte_s, k=args.k)
    inhole_s = masking.region_split(Xtr_s, removed_s, Xte_s, k=args.k)

    # --- train baseline + the two masked models, predict on the fixed test set ---
    def fit_predict(kept):
        m = _make_model(model_name, seed=config.RANDOM_SEED + args.seed)
        m.fit(Xtr[kept], ytr[kept])
        return m.predict(Xte)
    print("  training baseline ...", flush=True)
    pred_base = fit_predict(np.arange(n_train))
    print("  training uniform-masked ...", flush=True)
    pred_u = fit_predict(kept_u)
    print("  training structured-masked ...", flush=True)
    pred_s = fit_predict(kept_s)

    rmse_base = metrics.rmse(yte, pred_base)
    rmse_u = metrics.rmse(yte, pred_u)
    rmse_s = metrics.rmse(yte, pred_s)
    err_base = np.abs(yte - pred_base)
    err_u = np.abs(yte - pred_u)
    err_s = np.abs(yte - pred_s)

    # subsample for drawing (geometry/scale all computed on full data above)
    tr_idx = _subsample(n_train, args.max_points, seed=1)
    te_idx = _subsample(len(yte), args.max_points, seed=2)

    # ============================ figure ============================
    fig, axes = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)
    xlo, xhi = np.percentile(Ptr[:, 0], [0.5, 99.5])
    ylo, yhi = np.percentile(Ptr[:, 1], [0.5, 99.5])

    def style_xy(ax):
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel(ax_lbl[0]); ax.set_ylabel(ax_lbl[1])

    # ---- Row 1: DATA (kept vs removed) ----
    def data_panel(ax, kept, removed, title):
        rem_mask = np.zeros(n_train, dtype=bool); rem_mask[removed] = True
        keep_mask = ~rem_mask
        ki = tr_idx[keep_mask[tr_idx]]
        ri = tr_idx[rem_mask[tr_idx]]
        ax.scatter(Ptr[ri, 0], Ptr[ri, 1], s=8, c="0.8", alpha=0.5, label="removed", zorder=1)
        ax.scatter(Ptr[ki, 0], Ptr[ki, 1], s=8, c="#1f77b4", alpha=0.6, label="kept", zorder=2)
        ax.set_title(title); style_xy(ax)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    data_panel(axes[0, 0], np.arange(n_train), np.array([], int),
               f"DATA  original train  (n={n_train})")
    data_panel(axes[0, 1], kept_u, removed_u,
               f"DATA  uniform mask  p={p}  (kept {len(kept_u)}, removed {len(removed_u)})")
    data_panel(axes[0, 2], kept_s, removed_s,
               f"DATA  structured mask  p={p}  (kept {len(kept_s)}, removed {len(removed_s)})")

    # ---- Row 2: RESULT error map (shared colour scale) ----
    vmax = float(np.percentile(np.concatenate([err_base, err_u, err_s]), 95)) or 1.0

    def err_panel(ax, err, title, inhole=None):
        sc = ax.scatter(Pte[te_idx, 0], Pte[te_idx, 1], c=err[te_idx],
                        cmap="inferno_r", vmin=0, vmax=vmax, s=12, alpha=0.85, zorder=2)
        if inhole is not None:
            ih = te_idx[inhole[te_idx]]
            ax.scatter(Pte[ih, 0], Pte[ih, 1], s=34, facecolors="none",
                       edgecolors="#00b894", linewidths=0.7, zorder=3,
                       label=f"in-hole test ({int(inhole.sum())})")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title(title); style_xy(ax)
        return sc

    err_panel(axes[1, 0], err_base, f"RESULT  baseline  |err|   RMSE={rmse_base:.3g}")
    err_panel(axes[1, 1], err_u, f"RESULT  uniform  |err|   RMSE={rmse_u:.3g}", inhole_u)
    sc = err_panel(axes[1, 2], err_s, f"RESULT  structured  |err|   RMSE={rmse_s:.3g}", inhole_s)
    cb = fig.colorbar(sc, ax=axes[1, :].tolist(), shrink=0.9, location="right")
    cb.set_label("|y - y_hat|  (shared scale)")

    # ---- Row 3: RESULT predicted vs actual ----
    lo = float(min(yte.min(), pred_base.min(), pred_u.min(), pred_s.min()))
    hi = float(max(yte.max(), pred_base.max(), pred_u.max(), pred_s.max()))

    def fit_panel(ax, pred, rmse, title, color):
        ax.scatter(yte[te_idx], pred[te_idx], s=8, alpha=0.4, c=color)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("actual y"); ax.set_ylabel("predicted y")
        ax.set_title(f"{title}   RMSE={rmse:.3g}")
        ax.text(0.04, 0.92, f"RMSE={rmse:.3g}", transform=ax.transAxes,
                fontsize=9, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fit_panel(axes[2, 0], pred_base, rmse_base, "FIT  baseline", "#636e72")
    fit_panel(axes[2, 1], pred_u, rmse_u, "FIT  uniform", "#0984e3")
    fit_panel(axes[2, 2], pred_s, rmse_s, "FIT  structured", "#d63031")

    gap = rmse_s - rmse_u
    fig.suptitle(
        f"{key} — masking effect ({model_name}, p={p}):  "
        f"baseline {rmse_base:.3g}  →  uniform {rmse_u:.3g}  →  structured {rmse_s:.3g}   "
        f"(unevenness penalty = {gap:+.3g})",
        fontsize=14)

    out_dir = os.path.join(config.OUTPUTS_DIR, "masking", "viz")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{key}__{model_name}__p{p}__viz.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  RMSE  baseline={rmse_base:.5g}  uniform={rmse_u:.5g}  structured={rmse_s:.5g}  "
          f"gap={gap:+.5g}", flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
