"""Full-data (0% hole) ceiling baseline — M00.

The in-flight E11 run predates the M00 edit, so it lacks the full-data reference needed to answer
"does any guided student beat a model trained on ALL the data?". This trains ONLY that model
(1 per dataset/seed, ~1/8 the cost of re-running E11), using the SAME config/seeds/binarizer/
y-encoding as E11 so the numbers are paired and directly comparable. Writes rmse_M00 per
(dataset, seed); compare_references.py / correlate_teacher_edge.py merge it automatically.

Usage:  python full_baseline.py [--datasets ccpp airquality california energy] [--seeds 6]
Out:    outputs/distillation/exp11/full_baseline.csv  (dataset, seed, rmse_M00)
"""
import argparse, csv, logging, os, sys, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.preprocessing import StandardScaler

from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor_v4 import TMRegressor   # same code path as E11

import config
from common import datasets, metrics
from common.data import get_split_data


def train_full(Xb, y, epochs, g_min, g_max, seed):
    """Plain student on the FULL training set, no teacher — identical harness to E11's train_student."""
    tm = TMRegressor(config.TM_NUM_CLAUSES, config.TM_T, config.TM_S, platform="CPU",
                     weighted_clauses=config.TM_WEIGHTED_CLAUSES, seed=seed)
    for ep in range(epochs):
        tm.fit(Xb, y, global_y_min=g_min, global_y_max=g_max)
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ccpp", "airquality", "california", "energy"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "distillation", "exp11"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for key in args.datasets:
        Xtr, Xte, ytr, yte = get_split_data(key)
        binr = StandardBinarizer(max_bits_per_feature=config.TM_MAX_BITS_PER_FEATURE)
        Xb, Xb_te = binr.fit_transform(Xtr).astype(np.uint32), binr.transform(Xte).astype(np.uint32)
        g_min, g_max = float(ytr.min()), float(ytr.max())
        print(f"\n=== {key}: {len(ytr)} train (FULL) / {len(yte)} test ===", flush=True)
        for seed in [config.RANDOM_SEED + i for i in range(args.seeds)]:
            M00 = train_full(Xb, ytr, args.epochs, g_min, g_max, seed)
            r = round(metrics.rmse(yte, M00.predict(Xb_te)), 5)
            rows.append(dict(dataset=key, seed=seed, rmse_M00=r))
            print(f"  seed {seed}: full RMSE = {r:.4f}", flush=True)

    out = os.path.join(args.out, "full_baseline.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "seed", "rmse_M00"]); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
