"""RTM control for experiment 7: is the err_unguided<->teacher_help correlation a
real 'teacher fixes hard points' effect, or just regression-to-the-mean?

Per seed (same student mask): train unguided_A, unguided_B (different model seed,
same data), and guided (teacher-distilled, shares seed with A). On the test set:
  teacher_help = |err_A| - |err_guided|     (A vs its guided counterpart)
  placebo_help = |err_A| - |err_B|          (A vs an independent unguided -> pure RTM)
Compare their correlation with |err_A| and their mean in the hardest error decile.
If teacher >> placebo, the teacher genuinely targets hard points beyond RTM.
"""
import sys, os, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
import config
from common.data import get_split_data
from experiment5_distillation import nested_structured, train_tm

KEY = sys.argv[sys.argv.index("--dataset")+1] if "--dataset" in sys.argv else "ccpp"
SEEDS = [config.RANDOM_SEED + i for i in range(5)]
EP = 100

Xtr, Xte, ytr, yte = get_split_data(KEY)
binr = StandardBinarizer(max_bits_per_feature=config.TM_MAX_BITS_PER_FEATURE)
Xb, Xb_te = binr.fit_transform(Xtr).astype(np.uint32), binr.transform(Xte).astype(np.uint32)
Xs = StandardScaler().fit_transform(Xtr)
g_min, g_max, ysd, n = float(ytr.min()), float(ytr.max()), float(np.std(ytr)) or 1.0, len(ytr)

errA, helpT, helpP = [], [], []
for s in SEEDS:
    _, rem_s = nested_structured(Xs, [0.2, 0.3], s, hole_frac=0.05)
    rem_t, _ = nested_structured(Xs, [0.2, 0.3], s, hole_frac=0.05)  # same call -> teacher removed (0.2)
    kept_s = np.setdiff1d(np.arange(n), rem_s)
    kept_t = np.setdiff1d(np.arange(n), rem_t)
    M20 = train_tm(Xb[kept_t], ytr[kept_t], EP, g_min, g_max, s)
    uA = train_tm(Xb[kept_s], ytr[kept_s], EP, g_min, g_max, s)
    uB = train_tm(Xb[kept_s], ytr[kept_s], EP, g_min, g_max, s + 1000)
    g = train_tm(Xb[kept_s], ytr[kept_s], EP, g_min, g_max, s, teacher=M20, f=0.75, f_opposite=0.0, p_start=0.5)
    eA, eB, eg = np.abs(yte - uA.predict(Xb_te)), np.abs(yte - uB.predict(Xb_te)), np.abs(yte - g.predict(Xb_te))
    errA.append(eA / ysd); helpT.append((eA - eg) / ysd); helpP.append((eA - eB) / ysd)

errA, helpT, helpP = np.concatenate(errA), np.concatenate(helpT), np.concatenate(helpP)
rt, _ = stats.spearmanr(errA, helpT); rp, _ = stats.spearmanr(errA, helpP)
hard = errA >= np.quantile(errA, 0.9)
print(f"{KEY}: n={len(errA)}")
print(f"  Spearman(|errA|, teacher_help)  = {rt:+.3f}")
print(f"  Spearman(|errA|, placebo_help)  = {rp:+.3f}   (RTM baseline)")
print(f"  mean help in hardest decile: teacher={helpT[hard].mean():+.4f}  placebo={helpP[hard].mean():+.4f}")
print(f"  mean help overall:           teacher={helpT.mean():+.4f}  placebo={helpP.mean():+.4f}")
print(f"  -> teacher beyond RTM (hardest decile): {helpT[hard].mean()-helpP[hard].mean():+.4f}")
