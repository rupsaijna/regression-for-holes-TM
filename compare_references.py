"""Compare every guided mode's test RMSE against the reference models present in an experiment's
per-dataset CSVs: unguided (no-teacher 30%-hole student), M20 (20%-hole teacher), and M00
(full-data 0%-hole ceiling, present from E11 on).

For each dataset x mode x reference: mean RMSE delta (reference - mode; >0 = mode BEATS reference)
and per-seed win count. Answers "does the guided student beat the teacher / the full-data model?"

Usage:  python compare_references.py [--exp-dir outputs/distillation/exp11]
"""
import argparse, csv, glob, os, sys
import numpy as np

REF_COLS = ["rmse_unguided", "rmse_M20", "rmse_M00"]   # references to compare against (if present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="outputs/distillation/exp11")
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(os.path.join(args.exp_dir, "*.csv")))
             if not p.endswith("_summary.csv") and not p.endswith("full_baseline.csv")]
    if not paths:
        sys.exit(f"No per-dataset CSVs in {args.exp_dir}")

    # optional external full-data ceiling (full_baseline.py), merged by (dataset, seed)
    full = {}
    fpath = os.path.join(args.exp_dir, "full_baseline.csv")
    if os.path.exists(fpath):
        for r in csv.DictReader(open(fpath, newline="")):
            full[(r["dataset"], r["seed"])] = r["rmse_M00"]
        print(f"(merged rmse_M00 from {os.path.basename(fpath)})")

    for path in paths:
        rows = list(csv.DictReader(open(path, newline="")))
        if not rows or "rmse_unguided" not in rows[0]:
            continue
        ds = rows[0]["dataset"]
        if "rmse_M00" not in rows[0] and full:                 # graft in the external ceiling
            for r in rows:
                if (r["dataset"], r["seed"]) in full:
                    r["rmse_M00"] = full[(r["dataset"], r["seed"])]
        refs = [c for c in REF_COLS if c in rows[0]]
        modes = [k[5:] for k in rows[0]
                 if k.startswith("rmse_") and k not in REF_COLS]
        n = len(rows)
        refvals = {c: np.array([float(r[c]) for r in rows]) for c in refs}
        print(f"\n=== {ds}  (n={n})  " +
              "  ".join(f"{c[5:]}={refvals[c].mean():.4f}" for c in refs) + " ===")
        hdr = f"  {'mode':18s} {'meanRMSE':>9s}" + "".join(f"  vs {c[5:]:>9s}" for c in refs)
        print(hdr)
        for m in modes:
            rg = np.array([float(r[f"rmse_{m}"]) for r in rows])
            cells = ""
            for c in refs:
                delta = refvals[c].mean() - rg.mean()      # >0 = mode beats reference
                wins = int((rg < refvals[c]).sum())
                mark = "*" if delta > 0 else " "
                cells += f"  {delta:+7.4f}{mark}{wins}/{n}"
            print(f"  {m:18s} {rg.mean():9.4f}{cells}")
    print("\n(* = mode beats that reference on the mean; x/n = seeds the mode wins)")


if __name__ == "__main__":
    main()
