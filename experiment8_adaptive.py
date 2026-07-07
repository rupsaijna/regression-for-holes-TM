"""Experiment 8 — ADAPTIVE teacher trigger.

E7 showed the teacher's genuine value concentrates on high-disagreement / high-error
points, not on geometry. So instead of firing the teacher correction with a uniform
per-sample probability p, fire it preferentially where the teacher and student DISAGREE
(or where the student errs). Budget-matched: per epoch, p_e = clip(p * d_e/mean(d_e), 0, 1)
so the EXPECTED number of fired samples ~ p*n (same as uniform) -- isolating *where* the
teacher acts, not *how much*.

Compares, on the masked-student task (teacher 20% -> student 30% mask, structured,
f=0.75, p=0.5, 100ep), held-out TEST, paired by seed:
  unguided  vs  guided-UNIFORM  vs  guided-ADAPTIVE(disagree)  vs  guided-ADAPTIVE(error)

Reports per dataset: improvement vs unguided for each, and adaptive_gain = adaptive - uniform.

Out: outputs/distillation/exp8/{<ds>__adaptive.csv, adaptive_summary.{csv,md}, adaptive__<ds>.png}
"""
import argparse, csv, logging, os, sys, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor

import config
from common import datasets, metrics
from common.data import get_split_data
from experiment5_distillation import nested_structured, teacher_encoded_predictions

METHODS = ["uniform", "adapt_disagree", "adapt_error"]


def encode_y(y, g_min, g_max, T):
    return np.clip(((np.asarray(y, float) - g_min) / (g_max - g_min) * T).round().astype(np.int64), 0, T)


def train_student(Xb, y, epochs, g_min, g_max, seed, teacher_vec=None, f=0.0, p=0.0,
                  adaptive=None, enc_y=None, T=config.TM_T):
    """adaptive in {None,'disagree','error'}; None -> uniform p (or no teacher if teacher_vec None)."""
    tm = TMRegressor(config.TM_NUM_CLAUSES, config.TM_T, config.TM_S, platform="CPU",
                     weighted_clauses=config.TM_WEIGHTED_CLAUSES, seed=seed)
    trng = np.random.RandomState(seed + 777)
    for ep in range(epochs):
        psamp = None
        if teacher_vec is not None and adaptive is not None and getattr(tm, "clause_bank", None) is not None:
            stu = teacher_encoded_predictions(tm, Xb)                  # student's current encoded preds
            d = np.abs(teacher_vec - stu).astype(float) if adaptive == "disagree" \
                else np.abs(stu - enc_y).astype(float)                 # 'error'
            md = d.mean()
            psamp = np.clip(p * d / md, 0.0, 1.0) if md > 0 else None
        tm.fit(Xb, y, teacher_pred_encoded=teacher_vec, f=f, f_opposite=0.0, p_teacher=p,
               teacher_p_sample=psamp, teacher_rng=trng, global_y_min=g_min, global_y_max=g_max)
    return tm


def run_dataset(key, args):
    Xtr, Xte, ytr, yte = get_split_data(key)
    binr = StandardBinarizer(max_bits_per_feature=config.TM_MAX_BITS_PER_FEATURE)
    Xb, Xb_te = binr.fit_transform(Xtr).astype(np.uint32), binr.transform(Xte).astype(np.uint32)
    Xs = StandardScaler().fit_transform(Xtr)
    g_min, g_max, T, n = float(ytr.min()), float(ytr.max()), config.TM_T, len(ytr)
    print(f"\n=== {key}: {n} train / {len(yte)} test ===", flush=True)

    rows = []
    for seed in [config.RANDOM_SEED + i for i in range(args.seeds)]:
        rem_t, rem_s = nested_structured(Xs, [0.2, 0.3], seed, hole_frac=0.05)
        kept_t, kept_s = np.setdiff1d(np.arange(n), rem_t), np.setdiff1d(np.arange(n), rem_s)
        Xb_s, y_s = Xb[kept_s], ytr[kept_s]
        M20 = train_student(Xb[kept_t], ytr[kept_t], args.epochs, g_min, g_max, seed)
        M30u = train_student(Xb_s, y_s, args.epochs, g_min, g_max, seed)
        tvec = teacher_encoded_predictions(M20, Xb_s)
        enc_y_s = encode_y(y_s, g_min, g_max, T)
        guided = {
            "uniform": train_student(Xb_s, y_s, args.epochs, g_min, g_max, seed, tvec, args.f, args.p_teacher),
            "adapt_disagree": train_student(Xb_s, y_s, args.epochs, g_min, g_max, seed, tvec, args.f,
                                            args.p_teacher, adaptive="disagree"),
            "adapt_error": train_student(Xb_s, y_s, args.epochs, g_min, g_max, seed, tvec, args.f,
                                         args.p_teacher, adaptive="error", enc_y=enc_y_s)}
        ru = metrics.rmse(yte, M30u.predict(Xb_te))
        rec = dict(dataset=key, seed=seed, rmse_unguided=round(ru, 5),
                   rmse_M20=round(metrics.rmse(yte, M20.predict(Xb_te)), 5))
        for m in METHODS:
            rg = metrics.rmse(yte, guided[m].predict(Xb_te))
            rec[f"rmse_{m}"] = round(rg, 5)
            rec[f"impr_{m}"] = round(ru - rg, 5)              # >0 = helps vs unguided
        rows.append(rec)
        print(f"  seed {seed}: ung={ru:.4f} | "
              + " ".join(f"{m}:{rec['impr_'+m]:+.3f}" for m in METHODS), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ccpp", "airquality", "california", "energy"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--f", type=float, default=0.75)
    ap.add_argument("--p-teacher", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "distillation", "exp8"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = []
    for key in args.datasets:
        rows = run_dataset(key, args)
        with open(os.path.join(args.out, f"{key}__adaptive.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        for m in METHODS:
            impr = np.array([r[f"impr_{m}"] for r in rows])
            t = float(np.mean(impr) / (np.std(impr) / np.sqrt(len(impr)))) if np.std(impr) > 0 else 0.0
            gain = impr - np.array([r["impr_uniform"] for r in rows])     # adaptive - uniform (paired)
            tg = float(np.mean(gain) / (np.std(gain) / np.sqrt(len(gain)))) if np.std(gain) > 0 else 0.0
            summary.append(dict(dataset=key, method=m, n=len(impr),
                                mean_impr=round(float(np.mean(impr)), 5), impr_t=round(t, 2),
                                mean_gain_vs_uniform=round(float(np.mean(gain)), 5),
                                gain_t=("-" if m == "uniform" else round(tg, 2))))
        _plot(key, rows, args.out)

    with open(os.path.join(args.out, "adaptive_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print("\n===== SUMMARY (impr = unguided - guided RMSE; gain = adaptive - uniform, paired) =====")
    print(f"  {'dataset':11s} {'method':15s} {'mean_impr':>9s} {'t':>6s} {'gain_vs_uni':>11s} {'gain_t':>7s}")
    for s in summary:
        print(f"  {s['dataset']:11s} {s['method']:15s} {s['mean_impr']:+9.4f} {s['impr_t']:+6.2f} "
              f"{s['mean_gain_vs_uniform']:+11.4f} {str(s['gain_t']):>7s}")


def _plot(key, rows, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = ["unguided"] + METHODS
    means = [0.0] + [np.mean([r[f"impr_{m}"] for r in rows]) for m in METHODS]
    stds = [0.0] + [np.std([r[f"impr_{m}"] for r in rows]) for m in METHODS]
    ax.bar(xs, means, yerr=stds, capsize=4,
           color=["#b2bec3"] + ["#0984e3" if x >= 0 else "#d63031" for x in means[1:]])
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("RMSE improvement vs unguided")
    ax.set_title(f"{key}: uniform vs adaptive teacher trigger"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"adaptive__{key}.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
