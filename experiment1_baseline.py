"""Experiment 1 -- Baseline.

Trains a RegressionTM (tmu) and a PyTorch MLP on the SAME fixed train/test split
of the California Housing dataset, reports average RMSE over 10 / 50 / 100 runs,
and plots a chosen slice of test samples (true value vs each model's prediction)
to a PNG.

Models can be saved (--save-model) and reloaded (--load-model) so the program
does not have to retrain to reproduce a visualization.

Examples
--------
  # quick smoke test (tiny TM, few epochs, 2 runs)
  python experiment1_baseline.py --quick --runs 2

  # full baseline: cumulative averages at 10, 50 and 100 runs, save the models
  python experiment1_baseline.py --runs 10 50 100 --save-model

  # reload saved models and just redraw a different slice
  python experiment1_baseline.py --load-model --slice 100:160 --out outputs/slice_100_160.png

Note on "runs": we train max(runs) models once and report the cumulative
mean +/- std at each requested checkpoint -- identical statistics to training
each count separately, but far cheaper.
"""
import argparse

import numpy as np

import config
from common.data import get_split_data
from common.metrics import rmse
from common import viz, datasets
from models.tm_model import TMModel
from models.nn_model import NNModel


def parse_slice(s, n):
    """Parse 'START:END[:STEP]' into a list of integer positions within [0, n)."""
    parts = s.split(":")
    if len(parts) == 1:  # single index
        start = int(parts[0]); return [start % n]
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else n
    step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return list(range(start, min(stop, n), step))


def report_cumulative(name, rmses, checkpoints):
    print(f"\n  {name} RMSE (cumulative over runs):")
    arr = np.asarray(rmses)
    for c in checkpoints:
        c = min(c, len(arr))
        sub = arr[:c]
        ci = 1.96 * sub.std() / np.sqrt(c) if c > 1 else 0.0
        print(f"    {c:4d} runs: mean={sub.mean():.4f}  std={sub.std():.4f}  95%CI=+/-{ci:.4f}")


def build_tm(args):
    if args.quick:
        return TMModel(num_clauses=50, T=400, s=args.s, epochs=3,
                       max_bits_per_feature=args.max_bits)
    return TMModel(num_clauses=args.clauses, T=args.T, s=args.s,
                   epochs=args.tm_epochs, max_bits_per_feature=args.max_bits)


def build_nn(args, seed):
    epochs = 10 if args.quick else args.nn_epochs
    return NNModel(epochs=epochs, seed=seed)


def main():
    p = argparse.ArgumentParser(description="Baseline: RegressionTM vs MLP on a regression dataset")
    p.add_argument("--dataset", default=datasets.DEFAULT_DATASET, choices=list(datasets.DATASETS),
                   help="Which registered dataset to run on (see --list-datasets)")
    p.add_argument("--list-datasets", action="store_true", help="Print available datasets and exit")
    p.add_argument("--runs", type=int, nargs="+", default=[10, 50, 100],
                   help="Run checkpoints for cumulative averaging, e.g. --runs 10 50 100")
    p.add_argument("--load-model", action="store_true", help="Load saved canonical models instead of training")
    p.add_argument("--save-model", action="store_true", help="Save run-0 models as the canonical models")
    p.add_argument("--slice", default="0:50", help="Test-sample slice for the plot, START:END[:STEP]")
    p.add_argument("--out", default=None, help="Output PNG path")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quick", action="store_true", help="Tiny/fast settings for a smoke test")
    p.add_argument("--no-progress", dest="progress", action="store_false",
                   help="Disable tqdm progress bars")
    p.set_defaults(progress=True)
    # hyperparameter overrides
    p.add_argument("--clauses", type=int, default=config.TM_NUM_CLAUSES)
    p.add_argument("--T", type=int, default=config.TM_T)
    p.add_argument("--s", type=float, default=config.TM_S)
    p.add_argument("--tm-epochs", type=int, default=config.TM_EPOCHS)
    p.add_argument("--nn-epochs", type=int, default=config.NN_EPOCHS)
    p.add_argument("--max-bits", type=int, default=config.TM_MAX_BITS_PER_FEATURE)
    args = p.parse_args()

    if args.list_datasets:
        print(datasets.describe_table())
        return

    meta = datasets.DATASETS[args.dataset]
    tm_path, nn_path = config.tm_model_file(args.dataset), config.nn_model_file(args.dataset)
    X_train, X_test, y_train, y_test = get_split_data(args.dataset)
    print(f"Dataset '{args.dataset}' ({meta['title']}): "
          f"{len(y_train)} train / {len(y_test)} test samples, {X_train.shape[1]} features")

    if args.load_model:
        print("Loading saved canonical models ...")
        tm, nn = TMModel.load(tm_path), NNModel.load(nn_path)
        tm_pred_test, nn_pred_test = tm.predict(X_test), nn.predict(X_test)
        print(f"  Loaded-model test RMSE:  RegressionTM={rmse(y_test, tm_pred_test):.4f}  "
              f"NeuralNet={rmse(y_test, nn_pred_test):.4f}")
        canonical_tm, canonical_nn = tm, nn
    else:
        checkpoints = sorted(set(args.runs))
        max_runs = max(checkpoints)
        print(f"Training {max_runs} run(s) of each model "
              f"({'QUICK' if args.quick else 'full'} settings) ...")
        tm_rmses, nn_rmses = [], []
        canonical_tm = canonical_nn = None
        from tqdm import tqdm
        run_bar = tqdm(range(max_runs), desc="runs", unit="run", disable=not args.progress)
        for r in run_bar:
            tm = build_tm(args).fit(X_train, y_train, progress=args.progress,
                                    desc=f"  TM run {r + 1}/{max_runs}")
            tm_rmses.append(rmse(y_test, tm.predict(X_test)))
            nn = build_nn(args, seed=config.RANDOM_SEED + r).fit(
                X_train, y_train, progress=args.progress, desc=f"  NN run {r + 1}/{max_runs}")
            nn_rmses.append(rmse(y_test, nn.predict(X_test)))
            if r == 0:
                canonical_tm, canonical_nn = tm, nn
            run_bar.set_postfix(TM=f"{tm_rmses[-1]:.4f}", NN=f"{nn_rmses[-1]:.4f}")
            tqdm.write(f"  run {r + 1:3d}/{max_runs}:  TM={tm_rmses[-1]:.4f}  NN={nn_rmses[-1]:.4f}")
        run_bar.close()

        report_cumulative("RegressionTM", tm_rmses, checkpoints)
        report_cumulative("NeuralNet", nn_rmses, checkpoints)

        if args.save_model:
            print(f"\nSaving canonical models -> {canonical_tm.save(tm_path)}\n"
                  f"                          {canonical_nn.save(nn_path)}")

    if args.no_plot:
        return

    positions = parse_slice(args.slice, len(y_test))
    pos = np.array(positions)
    y_slice = y_test[pos]
    preds = {
        "RegressionTM": canonical_tm.predict(X_test[pos]),
        "NeuralNet": canonical_nn.predict(X_test[pos]),
    }
    out = args.out or f"{config.OUTPUTS_DIR}/baseline_{args.dataset}_{positions[0]}_{positions[-1]}.png"
    viz.plot_predictions(positions, y_slice, preds,
                         title=f"{meta['title']} -- test samples [{args.slice}]",
                         out_path=out, ylabel=meta["target"])
    tm_r = rmse(y_slice, preds["RegressionTM"]); nn_r = rmse(y_slice, preds["NeuralNet"])
    print(f"\nPlot saved -> {out}")
    print(f"  Slice RMSE:  RegressionTM={tm_r:.4f}  NeuralNet={nn_r:.4f}  ({len(positions)} samples)")


if __name__ == "__main__":
    main()
