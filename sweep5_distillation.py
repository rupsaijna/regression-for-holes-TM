"""Experiment 5 — distillation sweep in the STABLE region (f_opposite=0).

Per (epochs, seed) the knob-independent models (M100, M20, per-student unguided) are
trained ONCE via build_shared; only the guided student varies across f. Reports val
RMSE (guided vs unguided vs M100) -- NOT gap_closed, whose denominator (unguided-M100)
is noise-unstable on TM.

Usage:
  python sweep5_distillation.py --dataset ccpp --seeds 6 --f-grid 0.5 0.75 1.0
"""
import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from experiment5_distillation import build_shared, train_guided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ccpp")
    ap.add_argument("--mode", default="structured", choices=["uniform", "structured"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--student-fracs", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--f-grid", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--teacher-frac", type=float, default=0.2)
    ap.add_argument("--f-opposite", type=float, default=0.0)
    ap.add_argument("--p-teacher", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "distillation"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seeds = [config.RANDOM_SEED + i for i in range(args.seeds)]
    rows = []
    for epochs in args.epochs:
        for seed in seeds:
            shared = build_shared(args.dataset, args.mode, seed, epochs,
                                  teacher_frac=args.teacher_frac, student_fracs=args.student_fracs)
            base = dict(dataset=args.dataset, mode=args.mode, epochs=epochs, seed=seed,
                        teacher_frac=args.teacher_frac, f_opposite=args.f_opposite, p_teacher=args.p_teacher,
                        rmse_M100=round(shared["rmse_M100"], 5), rmse_M20=round(shared["rmse_M20"], 5))
            print(f"[{args.dataset}/{args.mode}] ep={epochs} seed={seed}: "
                  f"M100={shared['rmse_M100']:.4f} M20={shared['rmse_M20']:.4f} "
                  + " ".join(f"u{sf}={shared['rmse_ung'][sf]:.4f}" for sf in args.student_fracs), flush=True)
            for sf in args.student_fracs:
                ru = shared["rmse_ung"][sf]
                # f=0 anchor (unguided)
                rows.append(dict(base, student_frac=sf, f=0.0, rmse_unguided=round(ru, 5),
                                 rmse_guided=round(ru, 5)))
                for f in args.f_grid:
                    rg, _ = train_guided(shared, sf, f, args.f_opposite, args.p_teacher)
                    rows.append(dict(base, student_frac=sf, f=f, rmse_unguided=round(ru, 5),
                                     rmse_guided=round(rg, 5)))
                    print(f"    sf={sf} f={f}: guided={rg:.4f} (unguided {ru:.4f})", flush=True)

    csv_path = os.path.join(args.out, f"sweep__{args.dataset}__{args.mode}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # ---- plot: guided RMSE vs f (f=0 == unguided), panel per student mask, line per epoch ----
    xs = [0.0] + list(args.f_grid)
    fig, axes = plt.subplots(1, len(args.student_fracs), figsize=(6.5 * len(args.student_fracs), 5), squeeze=False)
    for ax, sf in zip(axes[0], args.student_fracs):
        for epochs in args.epochs:
            means, stds = [], []
            for f in xs:
                vals = [r["rmse_guided"] for r in rows
                        if r["student_frac"] == sf and r["epochs"] == epochs and r["f"] == f]
                means.append(np.mean(vals)); stds.append(np.std(vals))
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=f"{epochs} epochs")
            m100 = np.mean([r["rmse_M100"] for r in rows if r["epochs"] == epochs])
            ax.axhline(m100, ls="--", lw=0.8, alpha=0.6,
                       color=ax.lines[-1].get_color())
        ax.set_title(f"{args.dataset}/{args.mode} — student {int(sf*100)}% (teacher {int(args.teacher_frac*100)}%)\n"
                     f"f_opposite={args.f_opposite}, p={args.p_teacher}, {args.seeds} seeds  "
                     f"(f=0 is unguided; dashed=M100)")
        ax.set_xlabel("same-side blend f"); ax.set_ylabel("val RMSE (mean±std over seeds)")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(args.out, f"sweep__{args.dataset}__{args.mode}.png")
    fig.savefig(png, dpi=120); plt.close(fig)
    print(f"plot -> {png}")

    # ---- compact summary (mean over seeds): guided-unguided delta ----
    print("\n  ep  studentMask    f    guidedRMSE   Δ(g-u) mean±std")
    for epochs in args.epochs:
        for sf in args.student_fracs:
            for f in xs:
                sel = [r for r in rows if r["epochs"] == epochs and r["student_frac"] == sf and r["f"] == f]
                g = np.array([r["rmse_guided"] for r in sel]); u = np.array([r["rmse_unguided"] for r in sel])
                d = g - u
                print(f"  {epochs:3d}     {int(sf*100):3d}%      {f:.2f}   {g.mean():8.4f}   "
                      f"{d.mean():+.4f} ± {d.std():.4f}")


if __name__ == "__main__":
    main()
