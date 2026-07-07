"""Dataset loading and the single, fixed train/test split.

Works for any dataset registered in common/datasets.py.  The split for a given
dataset is created once (deterministically, from RANDOM_SEED) and persisted to
splits/<dataset>.npz, so the baseline program and the analytics program always
operate on exactly the same rows for that dataset.

Preprocessing (binarization for the TM, standardization for the NN) is owned by
the individual model wrappers, not by this module -- here we only deal with the
raw feature matrix and the fixed split.  The one exception is `standardize()`,
a helper the analytics program uses to put distances on a common scale.
"""
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from common import datasets


def get_raw_data(dataset=datasets.DEFAULT_DATASET):
    """Return (X, Y, feature_names) for a registered dataset."""
    return datasets.load(dataset)


def get_split_indices(dataset=datasets.DEFAULT_DATASET):
    """Return (train_idx, test_idx), creating and persisting them on first use."""
    path = config.split_file(dataset)
    if os.path.exists(path):
        d = np.load(path)
        return d["train_idx"], d["test_idx"]

    _, Y, _ = get_raw_data(dataset)
    idx = np.arange(len(Y))
    train_idx, test_idx = train_test_split(
        idx, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, shuffle=True
    )
    np.savez(path, train_idx=train_idx, test_idx=test_idx)
    return train_idx, test_idx


def get_split_data(dataset=datasets.DEFAULT_DATASET):
    """Return raw (X_train, X_test, Y_train, Y_test) for the dataset's fixed split."""
    X, Y, _ = get_raw_data(dataset)
    train_idx, test_idx = get_split_indices(dataset)
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


def standardize(X_train, *others):
    """Fit a StandardScaler on X_train and apply it to X_train + any extra arrays.

    Used by the analytics program so that every distance is measured in the same
    z-scored feature space the models effectively operate in.
    Returns (scaler, X_train_scaled, *others_scaled).
    """
    scaler = StandardScaler().fit(X_train)
    out = [scaler.transform(X_train)] + [scaler.transform(o) for o in others]
    return (scaler, *out)
