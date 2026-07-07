"""RegressionTM (tmu) wrapper.

The wrapper owns its own binarizer, so it accepts *raw* feature matrices and a
saved model is fully self-contained: load it and call .predict(X_raw) directly.

tmu / pycuda imports are done lazily inside the methods so that this module can
be imported on a machine without tmu (e.g. just to inspect the code).
"""
import pickle

import numpy as np

import config


class TMModel:
    def __init__(self, num_clauses=config.TM_NUM_CLAUSES, T=config.TM_T, s=config.TM_S,
                 platform=config.TM_PLATFORM, weighted_clauses=config.TM_WEIGHTED_CLAUSES,
                 epochs=config.TM_EPOCHS, max_bits_per_feature=config.TM_MAX_BITS_PER_FEATURE):
        self.num_clauses = num_clauses
        self.T = T
        self.s = s
        self.platform = platform
        self.weighted_clauses = weighted_clauses
        self.epochs = epochs
        self.max_bits_per_feature = max_bits_per_feature
        self.binarizer = None
        self.tm = None

    def _binarize_fit(self, X):
        from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
        self.binarizer = StandardBinarizer(max_bits_per_feature=self.max_bits_per_feature)
        return self.binarizer.fit_transform(X).astype(np.uint32)

    def _binarize(self, X):
        return self.binarizer.transform(X).astype(np.uint32)

    def fit(self, X_train_raw, y_train, progress=False, desc="TM"):
        from tmu.models.regression.vanilla_regressor import TMRegressor
        Xb = self._binarize_fit(X_train_raw)
        y_train = np.asarray(y_train, dtype=np.float64)
        self.tm = TMRegressor(self.num_clauses, self.T, self.s,
                              platform=self.platform, weighted_clauses=self.weighted_clauses)
        iterator = range(self.epochs)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=desc, unit="epoch", leave=False)
        for epoch in iterator:
            self.tm.fit(Xb, y_train)
        return self

    def predict(self, X_raw):
        return np.asarray(self.tm.predict(self._binarize(X_raw)), dtype=np.float64)

    def fit_with_history(self, X_train_raw, y_train, X_eval_raw, y_eval, max_epochs=None):
        """Train like fit() but evaluate test RMSE after every epoch.

        tmu's TMRegressor.fit is incremental (one pass per call), so predicting
        between calls yields the convergence curve at no extra training cost.
        Returns the per-epoch eval-RMSE list; the fitted model is left ready for
        .predict(). `max_epochs` overrides self.epochs.
        """
        from tmu.models.regression.vanilla_regressor import TMRegressor
        epochs = int(max_epochs or self.epochs)
        Xb = self._binarize_fit(X_train_raw)
        y_train = np.asarray(y_train, dtype=np.float64)
        self.tm = TMRegressor(self.num_clauses, self.T, self.s,
                              platform=self.platform, weighted_clauses=self.weighted_clauses)
        Xev_b = self._binarize(X_eval_raw)
        y_eval = np.asarray(y_eval, dtype=np.float64)
        history = []
        for epoch in range(epochs):
            self.tm.fit(Xb, y_train)
            pred = np.asarray(self.tm.predict(Xev_b), dtype=np.float64)
            history.append(float(np.sqrt(np.mean((pred - y_eval) ** 2))))
        return history

    # --- persistence -------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
