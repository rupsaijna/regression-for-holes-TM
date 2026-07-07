"""Experiment 2 -- Data analytics.

Operates on the SAME fixed split as experiment1_baseline.py.

(A) Descriptive distance statistics for the train set, the test set and the
    whole dataset: spread, average / minimum / maximum pairwise distance.

(B) "Spacing" of every test point under ALL candidate definitions, computed in
    two feature spaces (standardized and PCA-2D):
        nn_to_train_k1   - distance to nearest training point
        knn_to_train_k5  - mean distance to 5 nearest training points
        local_density_k5 - mean distance to 5 nearest points in the whole set
        nn_to_test_k1    - distance to nearest other test point

(C) Model behaviour vs spacing.  Loads the canonical TM and NN models saved by
    experiment1 (--save-model) and, for each spacing definition, produces:
      * a per-slice diagnostic figure (predictions + spacing + spacing-vs-error)
      * a whole-test-set RMSE-per-spacing-bucket plot
      * Pearson / Spearman correlation between spacing and each model's abs error
    This is what tests the hypothesis "wider spacing -> poorer regression".

Run experiment1_baseline.py --save-model first so the models exist.

Example:
  python experiment2_analytics.py --slice 0:50 --buckets 8 --space both
"""
import argparse
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

import config
from common.data import get_split_data, standardize
from common import metrics, viz, datasets
from models.tm_model import TMModel
from models.nn_model import NNModel

MODELS = ("RegressionTM", "NeuralNet")


def descriptive_stats(X_train_s, X_test_s, X_all_s):
    print("\n" + "=" * 78)
    print("(A) DESCRIPTIVE DISTANCE STATISTICS  (standardized Euclidean)")
    print("=" * 78)
    print(f"{'set':6s} {'n':>7s} {'spread':>10s} {'avg_pair':>10s} {'min_pair':>10s} {'max_pair':>10s}  note")
    for label, X in (("train", X_train_s), ("test", X_test_s), ("whole", X_all_s)):
        s = metrics.distance_summary(X)
        note = "exact" if s["avg_max_exact"] else "avg/max sampled"
        print(f"{label:6s} {s['n']:7d} {s['spread']:10.4f} {s['avg_pairwise']:10.4f} "
              f"{s['min_pairwise']:10.4f} {s['max_pairwise']:10.4f}  ({note})")


def make_spaces(X_train, X_test):
    """Return dict {space_name: (X_train_s, X_test_s, X_all_s)} for distance work."""
    scaler, X_train_s, X_test_s = standardize(X_train, X_test)
    X_all_s = np.vstack([X_train_s, X_test_s])

    pca = PCA(n_components=2, random_state=config.RANDOM_SEED).fit(X_train_s)
    tr_p, te_p = pca.transform(X_train_s), pca.transform(X_test_s)
    all_p = np.vstack([tr_p, te_p])

    return {
        "standardized": (X_train_s, X_test_s, X_all_s),
        "PCA-2D": (tr_p, te_p, all_p),
    }


def parse_slice(s, n):
    """Parse 'START:END[:STEP]' into a list of integer positions within [0, n)."""
    parts = s.split(":")
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else n
    step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return list(range(start, min(stop, n), step))


def bucket_curves(spacing, y_true, preds_by_model, n_buckets):
    """For one spacing array, compute (bucket_centers, {model: rmse_per_bucket}).

    Buckets are spacing quantiles; emptiness depends only on the (shared) spacing
    array, so the centers are common to every model.
    """
    edges = np.quantile(spacing, np.linspace(0, 1, n_buckets + 1))
    edges[-1] += 1e-9
    centers = []
    masks = []
    for b in range(n_buckets):
        m = (spacing >= edges[b]) & (spacing < edges[b + 1])
        if m.sum() == 0:
            continue
        centers.append(float(np.median(spacing[m])))
        masks.append(m)
    rmse_by_model = {
        name: np.array([metrics.rmse(y_true[m], yp[m]) for m in masks])
        for name, yp in preds_by_model.items()
    }
    return np.array(centers), rmse_by_model


def main():
    p = argparse.ArgumentParser(description="Analytics: distances + model-error vs spacing")
    p.add_argument("--dataset", default=datasets.DEFAULT_DATASET, choices=list(datasets.DATASETS),
                   help="Which registered dataset to analyze (must match a saved baseline)")
    p.add_argument("--slice", default="0:50", help="Test-sample slice for diagnostic panels")
    p.add_argument("--buckets", type=int, default=8, help="Spacing quantile buckets for RMSE curves")
    p.add_argument("--space", choices=["standardized", "PCA-2D", "both"], default="both",
                   help="Feature space(s) for the per-slice diagnostic panels")
    p.add_argument("--out-dir", default=None,
                   help="Figure output dir (default: outputs/analytics/<dataset>)")
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(config.OUTPUTS_DIR, "analytics", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dataset '{args.dataset}' ({datasets.DATASETS[args.dataset]['title']})")
    X_train, X_test, y_train, y_test = get_split_data(args.dataset)
    spaces = make_spaces(X_train, X_test)

    # (A) descriptive stats in the standardized space
    Xtr_s, Xte_s, Xall_s = spaces["standardized"]
    descriptive_stats(Xtr_s, Xte_s, Xall_s)

    # (B) all spacings, in every space
    print("\n" + "=" * 78)
    print("(B) PER-TEST-POINT SPACING (summary; full arrays used below)")
    print("=" * 78)
    spacings = {}  # space -> {spacing_name: array}
    for space_name, (tr, te, al) in spaces.items():
        spacings[space_name] = metrics.all_spacings(te, tr, al)
        print(f"\n  [{space_name}]")
        for name, arr in spacings[space_name].items():
            print(f"    {name:18s} min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}")

    # (C) model behaviour vs spacing -- needs the saved canonical models
    tm_path, nn_path = config.tm_model_file(args.dataset), config.nn_model_file(args.dataset)
    if not (os.path.exists(tm_path) and os.path.exists(nn_path)):
        print("\n[!] Canonical models not found for this dataset. Run:")
        print(f"      python experiment1_baseline.py --dataset {args.dataset} --save-model")
        print("    then re-run this program for section (C).")
        return

    tm, nn = TMModel.load(tm_path), NNModel.load(nn_path)
    preds_full = {"RegressionTM": tm.predict(X_test), "NeuralNet": nn.predict(X_test)}
    abs_err = {m: np.abs(preds_full[m] - y_test) for m in MODELS}

    print("\n" + "=" * 78)
    print("(C) SPACING vs MODEL ERROR  (whole test set)")
    print("=" * 78)
    print(f"  Overall test RMSE:  RegressionTM={metrics.rmse(y_test, preds_full['RegressionTM']):.4f}  "
          f"NeuralNet={metrics.rmse(y_test, preds_full['NeuralNet']):.4f}")
    print(f"\n  Correlation of spacing with |error|  (Pearson / Spearman):")
    print(f"  {'space':16s} {'spacing':18s} {'RegressionTM':>22s} {'NeuralNet':>22s}")
    for space_name, sp in spacings.items():
        for sname, sarr in sp.items():
            cells = []
            for m in MODELS:
                pr = pearsonr(sarr, abs_err[m])[0]
                sr = spearmanr(sarr, abs_err[m])[0]
                cells.append(f"{pr:+.3f}/{sr:+.3f}")
            print(f"  {space_name:16s} {sname:18s} {cells[0]:>22s} {cells[1]:>22s}")

    # --- figures: per-slice diagnostic panel for every spacing -------------
    positions = parse_slice(args.slice, len(y_test))
    pos = np.array(positions)
    y_slice = y_test[pos]
    preds_slice = {m: preds_full[m][pos] for m in MODELS}

    panel_spaces = list(spaces.keys()) if args.space == "both" else [args.space]
    print(f"\n  Writing figures to {out_dir} ...")
    for space_name in panel_spaces:
        for sname, sarr in spacings[space_name].items():
            tag = space_name.replace("-", "").lower()
            out = os.path.join(out_dir, f"panel_{tag}_{sname}.png")
            viz.plot_spacing_panel(positions, y_slice, preds_slice, sarr[pos], sname,
                                   out_path=out, space_label=space_name)
            print(f"    panel  -> {out}")

    # --- figures: whole-test-set RMSE-per-spacing-bucket, per spacing -------
    for space_name, sp in spacings.items():
        for sname, sarr in sp.items():
            centers, rmse_by_model = bucket_curves(sarr, y_test, preds_full, args.buckets)
            tag = space_name.replace("-", "").lower()
            out = os.path.join(out_dir, f"buckets_{tag}_{sname}.png")
            viz.plot_spacing_buckets(centers, rmse_by_model, sname, out_path=out, space_label=space_name)
            print(f"    bucket -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
