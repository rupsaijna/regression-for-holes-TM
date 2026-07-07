"""Shared configuration for the Regression experiments.

Both experiment1_baseline.py and experiment2_analytics.py import from here so
that the dataset, the fixed train/test split and all hyperparameters are defined
in exactly one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for _d in (SPLITS_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Fixed split -----------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
SPLIT_FILE = os.path.join(SPLITS_DIR, "split.npz")

# --- Tsetlin Machine (RegressionTM) ---------------------------------------
# Defaults follow tmu's RegressionDemo.py.
TM_NUM_CLAUSES = 1000
TM_T = 5000
TM_S = 2.75
TM_PLATFORM = "CPU"
TM_WEIGHTED_CLAUSES = True
TM_EPOCHS = 30
TM_MAX_BITS_PER_FEATURE = 10

# --- Neural network (PyTorch MLP) -----------------------------------------
NN_HIDDEN = (64, 64)
NN_LR = 1e-3
NN_EPOCHS = 100
NN_BATCH_SIZE = 128
NN_WEIGHT_DECAY = 0.0

# --- Per-dataset file paths -----------------------------------------------
# Each dataset gets its own fixed split and its own canonical saved models, so
# experiments on different datasets never clobber each other.
def split_file(dataset):
    return os.path.join(SPLITS_DIR, f"{dataset}.npz")


def tm_model_file(dataset):
    return os.path.join(MODELS_DIR, f"{dataset}__tm.pkl")


def nn_model_file(dataset):
    return os.path.join(MODELS_DIR, f"{dataset}__nn.pt")
