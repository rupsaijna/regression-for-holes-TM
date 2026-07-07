#!/usr/bin/env bash
# Experiment-5 effectiveness run, focused config (f=0.75, f_opp=0, p=0.5, student 0.3,
# 100ep, 5 seeds), SMALL/medium datasets. Big ones (metro beijing nyse nyse_log) run
# separately once the ccpp/california sweeps free up CPU.
set -u
cd /mnt/c/Users/Trshant/Rupsa/TMExperiments/RegressionExperiment
PY=~/regexp-venv/bin/python
for ds in autompg realestate mortality bloodfat airquality; do
  echo "===== $ds ($(date +%T)) ====="
  $PY sweep5_distillation.py --dataset "$ds" --mode structured --seeds 5 --epochs 100 \
      --student-fracs 0.3 --f-grid 0.75 2>/dev/null
  echo "$ds exit=$?"
done
echo "ALL SMALL DONE"
