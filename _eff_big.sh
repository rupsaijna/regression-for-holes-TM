#!/usr/bin/env bash
# Experiment-5 effectiveness, focused config, BIG datasets (run sequentially to limit
# CPU contention with the ccpp/california sweeps). nyse/nyse_log are the key heavy-tail
# test points for the density hypothesis.
set -u
cd /mnt/c/Users/Trshant/Rupsa/TMExperiments/RegressionExperiment
PY=~/regexp-venv/bin/python
for ds in beijing metro nyse_log nyse; do
  echo "===== $ds ($(date +%T)) ====="
  $PY sweep5_distillation.py --dataset "$ds" --mode structured --seeds 5 --epochs 100 \
      --student-fracs 0.3 --f-grid 0.75 2>/dev/null
  echo "$ds exit=$?"
done
echo "ALL BIG DONE"
