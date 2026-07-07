"""Point-1 sanity check for the teacher-guided RegressionTM (experiment 5).

Validates, BEFORE any compute-heavy sweep, that the new teacher path in
tmu/.../vanilla_regressor.py::fit is correct:

  (A) determinism      : two plain trainings at the same seed are identical.
  (B) off == baseline  : a student WITH a teacher attached but the correction
                         disabled is bit-identical to the unguided student, via
                         BOTH "off" routes -- p_teacher=0, and f=f_opposite=0.
  (C) on != baseline   : with the correction active (f>0, p_teacher>0) the
                         student's predictions actually change.

Run: wsl ~/regexp-venv/bin/python distill_sanity.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor

from common.data import get_split_data
from experiment5_distillation import teacher_encoded_predictions

KEY = "ccpp"
N_TRAIN = 1500      # small subset -> fast sanity
N_EVAL = 500
EPOCHS = 3
SEED = 1
CLAUSES, T, S = 200, 5000, 2.75


def make_data():
    Xtr, Xte, ytr, yte = get_split_data(KEY)
    b = StandardBinarizer(max_bits_per_feature=10)
    Xb_tr = b.fit_transform(Xtr).astype(np.uint32)
    Xb_te = b.transform(Xte).astype(np.uint32)
    g_min, g_max = float(ytr.min()), float(ytr.max())
    return (Xb_tr[:N_TRAIN], ytr[:N_TRAIN], Xb_te[:N_EVAL], yte[:N_EVAL], g_min, g_max)


def train(Xb, y, g_min, g_max, teacher=None, f=0.0, f_opposite=0.0, p_teacher=0.0):
    tm = TMRegressor(CLAUSES, T, S, platform="CPU", weighted_clauses=True, seed=SEED)
    trng = np.random.RandomState(123)
    tpe = teacher_encoded_predictions(teacher, Xb) if teacher is not None else None
    for _ in range(EPOCHS):
        tm.fit(Xb, y, teacher_pred_encoded=tpe,
               f=f, f_opposite=f_opposite, p_teacher=p_teacher, teacher_rng=trng,
               global_y_min=g_min, global_y_max=g_max)
    return tm


def main():
    Xb, y, Xb_e, y_e, g_min, g_max = make_data()
    print(f"sanity on {KEY}: {len(y)} train / {len(y_e)} eval, {EPOCHS} epochs, seed={SEED}\n")

    # (A) determinism
    a = train(Xb, y, g_min, g_max).predict(Xb_e)
    b = train(Xb, y, g_min, g_max).predict(Xb_e)
    d_det = float(np.max(np.abs(a - b)))
    print(f"(A) determinism            max|Δpred| = {d_det:.3e}   -> {'OK' if d_det == 0 else 'NON-DETERMINISTIC'}")

    # a teacher to attach (its identity doesn't matter for the off-tests)
    teacher = train(Xb, y, g_min, g_max)

    # (B) off == baseline, route 1: p_teacher = 0
    base = a
    off1 = train(Xb, y, g_min, g_max, teacher=teacher, f=0.5, f_opposite=1.0, p_teacher=0.0).predict(Xb_e)
    d_off1 = float(np.max(np.abs(base - off1)))
    print(f"(B1) off via p_teacher=0    max|Δpred| = {d_off1:.3e}   -> {'OK' if d_off1 == 0 else 'FAIL'}")

    # (B) off == baseline, route 2: f = f_opposite = 0  (correction fires but blends nothing)
    off2 = train(Xb, y, g_min, g_max, teacher=teacher, f=0.0, f_opposite=0.0, p_teacher=1.0).predict(Xb_e)
    d_off2 = float(np.max(np.abs(base - off2)))
    print(f"(B2) off via f=f_opp=0      max|Δpred| = {d_off2:.3e}   -> {'OK' if d_off2 == 0 else 'FAIL'}")

    # (C) on != baseline
    on = train(Xb, y, g_min, g_max, teacher=teacher, f=0.5, f_opposite=1.0, p_teacher=1.0).predict(Xb_e)
    d_on = float(np.max(np.abs(base - on)))
    print(f"(C) on (f=.5,p=1)           max|Δpred| = {d_on:.3e}   -> {'OK (changed)' if d_on > 0 else 'NO EFFECT'}")

    ok = (d_det == 0 and d_off1 == 0 and d_off2 == 0 and d_on > 0)
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
