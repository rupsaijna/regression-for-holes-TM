"""Experiment 5 — teacher-guided (distillation) RegressionTM.

Setup (see memory note regexp-experiment5-distillation):
  base b        = training pool (after carving a validation set out of the train split)
  teacher M20   = b with teacher_frac masked (default 20% -> 80% kept)  [stronger TEACHER]
  student       = b with student_frac masked (>= teacher_frac, masks NESTED ⊂ teacher) [weaker]
  M100          = trained on all of b (full-data reference)

While training the guided student, per sample with prob p_teacher we blend the teacher's
ENCODED prediction error into the student's (same-side -> fraction f; opposite-side ->
fraction f_opposite, 1=hard replace -- NOTE: f_opposite≈1 is unstable / diverges). We compare,
on the carved val set, student(unguided) vs student(guided) vs M20 vs M100, and report:
  gap_closed = (RMSE_unguided - RMSE_guided) / (RMSE_unguided - RMSE_M100)

All models share ONE binarizer (fit on b) + a GLOBAL y-encoding range so the encoded errors
are comparable. Off (f=f_opposite=0 or p_teacher=0) == stock training (see distill_sanity.py).

build_shared() trains the knob-independent models (M100, M20, per-student unguided) once;
train_guided() trains only the guided student -- so sweeps don't re-train the baselines.

Usage:
  python experiment5_distillation.py --dataset ccpp --mode structured --epochs 30 \
         --f 0.5 --f-opposite 0.0 --p-teacher 0.5
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor

import config
from common import datasets, metrics
from common.data import get_split_data


# --------------------------------------------------------------------------- #
# Nested masks: removed(small frac) ⊂ removed(large frac) so each student's kept
# set is a strict subset of the teacher's. Return removed-index arrays (ascending).
# --------------------------------------------------------------------------- #
def nested_uniform(n, fracs, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[:int(round(f * n))]) for f in fracs]


def nested_structured(X_std, fracs, seed, hole_frac=0.05):
    n = len(X_std)
    hole_size = max(1, int(round(hole_frac * n)))
    nn = NearestNeighbors(n_neighbors=min(hole_size, n)).fit(X_std)
    rng = np.random.default_rng(seed)
    removed, snaps = set(), []
    for f in fracs:                                   # cumulative carve -> nested
        target = int(round(f * n))
        while len(removed) < target:
            survivors = np.fromiter((i for i in range(n) if i not in removed), dtype=int)
            sp = int(rng.choice(survivors))
            order = nn.kneighbors(X_std[sp:sp + 1], return_distance=False)[0]
            for j in order:
                if len(removed) >= target:
                    break
                removed.add(int(j))
        snaps.append(np.sort(np.fromiter(removed, dtype=int)))
    return snaps


def _anneal(p_start, p_end, epoch, epochs):
    if epochs <= 1 or p_end == p_start:
        return p_start
    return p_start + (p_end - p_start) * (epoch / (epochs - 1))


def teacher_encoded_predictions(teacher, Xb):
    """Frozen teacher's ENCODED predictions clip(dot(w, clause_outputs), 0, T) for
    every row of Xb. Computed once, in isolation (no interleaving with a student's
    clause-bank updates), so the teacher signal is deterministic / order-independent."""
    enc = teacher.clause_bank.prepare_X(Xb)
    w = teacher.weight_bank.get_weights()
    preds = np.empty(len(Xb), dtype=np.int64)
    for e in range(len(Xb)):
        co = teacher.clause_bank.calculate_clause_outputs_predict(enc, e)
        preds[e] = int(np.clip(np.dot(w, co).astype(np.int32), 0, teacher.T))
    return preds


def train_tm(Xb, y, epochs, g_min, g_max, seed,
             teacher=None, f=0.0, f_opposite=0.0, p_start=0.0, p_end=None):
    """Train a RegressionTM for `epochs`, annealing p_teacher from p_start->p_end."""
    tm = TMRegressor(config.TM_NUM_CLAUSES, config.TM_T, config.TM_S,
                     platform="CPU", weighted_clauses=config.TM_WEIGHTED_CLAUSES, seed=seed)
    teacher_rng = np.random.RandomState(seed + 777)
    teacher_pred_encoded = teacher_encoded_predictions(teacher, Xb) if teacher is not None else None
    if p_end is None:
        p_end = p_start
    for ep in range(epochs):
        p_now = _anneal(p_start, p_end, ep, epochs)
        tm.fit(Xb, y, teacher_pred_encoded=teacher_pred_encoded,
               f=f, f_opposite=f_opposite, p_teacher=p_now, teacher_rng=teacher_rng,
               global_y_min=g_min, global_y_max=g_max)
    return tm


# --------------------------------------------------------------------------- #
# Knob-independent models (trained once, reused across a knob sweep).
# --------------------------------------------------------------------------- #
def build_shared(key, mode, seed, epochs, teacher_frac=0.2, student_fracs=(0.3,),
                 val_frac=0.15, hole_frac=0.05, max_bits=config.TM_MAX_BITS_PER_FEATURE):
    Xtr, Xte, ytr, yte = get_split_data(key)
    # carve a fixed validation set (same across seeds); b = remaining training pool
    Xb_raw, Xval_raw, y_core, y_val = train_test_split(
        Xtr, ytr, test_size=val_frac, random_state=config.RANDOM_SEED)

    binr = StandardBinarizer(max_bits_per_feature=max_bits)
    Xb = binr.fit_transform(Xb_raw).astype(np.uint32)
    Xb_val = binr.transform(Xval_raw).astype(np.uint32)
    g_min, g_max = float(y_core.min()), float(y_core.max())
    n = len(y_core)

    fracs = sorted(set([teacher_frac, *student_fracs]))          # ascending -> nested
    if mode == "uniform":
        removed_list = nested_uniform(n, fracs, seed)
    else:
        removed_list = nested_structured(StandardScaler().fit_transform(Xb_raw), fracs, seed, hole_frac)
    kept = {f: np.setdiff1d(np.arange(n), rem) for f, rem in zip(fracs, removed_list)}

    M100 = train_tm(Xb, y_core, epochs, g_min, g_max, seed)
    M20 = train_tm(Xb[kept[teacher_frac]], y_core[kept[teacher_frac]], epochs, g_min, g_max, seed)
    M_ung = {sf: train_tm(Xb[kept[sf]], y_core[kept[sf]], epochs, g_min, g_max, seed)
             for sf in student_fracs}

    def rmse(m):
        return float(metrics.rmse(y_val, m.predict(Xb_val)))

    return dict(
        key=key, mode=mode, seed=seed, epochs=epochs, teacher_frac=teacher_frac,
        Xb=Xb, y_core=y_core, Xb_val=Xb_val, y_val=y_val, g_min=g_min, g_max=g_max,
        kept=kept, M100=M100, M20=M20, M_ung=M_ung,
        rmse_M100=rmse(M100), rmse_M20=rmse(M20),
        rmse_ung={sf: rmse(M_ung[sf]) for sf in student_fracs})


def train_guided(shared, student_frac, f, f_opposite, p_start, p_end=None):
    """Train ONLY the guided student against the cached teacher; return (rmse, gap_closed)."""
    kept = shared["kept"][student_frac]
    m = train_tm(shared["Xb"][kept], shared["y_core"][kept], shared["epochs"],
                 shared["g_min"], shared["g_max"], shared["seed"],
                 teacher=shared["M20"], f=f, f_opposite=f_opposite, p_start=p_start, p_end=p_end)
    rg = float(metrics.rmse(shared["y_val"], m.predict(shared["Xb_val"])))
    ru, r100 = shared["rmse_ung"][student_frac], shared["rmse_M100"]
    denom = ru - r100
    gap = (ru - rg) / denom if denom != 0 else float("nan")
    return rg, gap


def run_once(key, mode, epochs, f, f_opposite, p_start, p_end, seed,
             val_frac=0.15, fractions=(0.2, 0.3), hole_frac=0.05):
    teacher_frac, student_frac = fractions
    shared = build_shared(key, mode, seed, epochs, teacher_frac=teacher_frac,
                          student_fracs=(student_frac,), val_frac=val_frac, hole_frac=hole_frac)
    rg, gap = train_guided(shared, student_frac, f, f_opposite, p_start, p_end)
    return dict(dataset=key, mode=mode, epochs=epochs, seed=seed,
                f=f, f_opposite=f_opposite, p_start=p_start, p_end=p_end,
                n_core=len(shared["y_core"]), n_kept_teacher=len(shared["kept"][teacher_frac]),
                n_kept_student=len(shared["kept"][student_frac]), n_val=len(shared["y_val"]),
                rmse_M100=round(shared["rmse_M100"], 6), rmse_M20=round(shared["rmse_M20"], 6),
                rmse_M30_unguided=round(shared["rmse_ung"][student_frac], 6),
                rmse_M30_guided=round(rg, 6), gap_closed=round(gap, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ccpp", choices=list(datasets.DATASETS))
    ap.add_argument("--mode", default="structured", choices=["uniform", "structured"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--f", type=float, default=0.5)
    ap.add_argument("--f-opposite", type=float, default=0.0)
    ap.add_argument("--p-teacher", type=float, default=0.5, help="p_start (= p_end unless --p-teacher-end)")
    ap.add_argument("--p-teacher-end", type=float, default=None, help="anneal target; default = constant")
    ap.add_argument("--repeats", type=int, default=1, help="seeds averaged (config.RANDOM_SEED + r)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--fractions", type=float, nargs=2, default=[0.2, 0.3],
                    help="teacher_mask student_mask (student >= teacher for nesting)")
    ap.add_argument("--hole-frac", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "distillation"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for r in range(args.repeats):
        row = run_once(args.dataset, args.mode, args.epochs, args.f, args.f_opposite,
                       args.p_teacher, args.p_teacher_end, config.RANDOM_SEED + r,
                       val_frac=args.val_frac, fractions=tuple(args.fractions), hole_frac=args.hole_frac)
        rows.append(row)
        print(f"  seed {config.RANDOM_SEED + r}: M100={row['rmse_M100']}  M20={row['rmse_M20']}  "
              f"M30u={row['rmse_M30_unguided']}  M30g={row['rmse_M30_guided']}  "
              f"gap_closed={row['gap_closed']}", flush=True)

    def mean(k):
        return float(np.mean([r[k] for r in rows]))
    print(f"\n=== {args.dataset} / {args.mode} / {args.epochs}ep | "
          f"f={args.f} f_opp={args.f_opposite} p={args.p_teacher}"
          f"{'->' + str(args.p_teacher_end) if args.p_teacher_end is not None else ''} | "
          f"repeats={args.repeats} ===")
    print(f"  RMSE  M100={mean('rmse_M100'):.4f}  M20(teacher)={mean('rmse_M20'):.4f}  "
          f"M30 unguided={mean('rmse_M30_unguided'):.4f}  M30 guided={mean('rmse_M30_guided'):.4f}")
    print(f"  gap_closed (toward M100) = {mean('gap_closed'):+.3f}   (>0 helps, <0 hurts; 1.0 = recovered)")

    tag = f"{args.dataset}__{args.mode}__{args.epochs}ep__f{args.f}_fo{args.f_opposite}_p{args.p_teacher}"
    csv_path = os.path.join(args.out, f"{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
