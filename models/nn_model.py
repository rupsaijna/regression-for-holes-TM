"""PyTorch MLP regressor wrapper.

Mirrors the TMModel interface: owns its own StandardScaler, accepts raw feature
matrices, and saves to a single self-contained file (state_dict + scaler stats +
architecture/config).
"""
import numpy as np
import torch
import torch.nn as nn

import config


class _MLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NNModel:
    def __init__(self, hidden=config.NN_HIDDEN, lr=config.NN_LR, epochs=config.NN_EPOCHS,
                 batch_size=config.NN_BATCH_SIZE, weight_decay=config.NN_WEIGHT_DECAY, seed=None):
        self.hidden = tuple(hidden)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.net = None
        # Feature StandardScaler stats (fit on train), stored as plain arrays
        self._mu = None
        self._sigma = None
        # Target standardization stats -- essential when the target magnitude is
        # large (e.g. CCPP ~450, mortality ~1000); the MLP trains on a z-scored
        # target and predictions are inverse-transformed back.
        self._ymu = 0.0
        self._ysigma = 1.0

    def _scale_fit(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        return (X - self._mu) / self._sigma

    def _scale(self, X):
        return (X - self._mu) / self._sigma

    def fit(self, X_train_raw, y_train, progress=False, desc="NN"):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        Xs = self._scale_fit(np.asarray(X_train_raw, dtype=np.float64))
        y_raw = np.asarray(y_train, dtype=np.float64)
        self._ymu = float(y_raw.mean())
        self._ysigma = float(y_raw.std()) or 1.0
        X = torch.tensor(Xs, dtype=torch.float32)
        y = torch.tensor((y_raw - self._ymu) / self._ysigma, dtype=torch.float32)

        self.net = _MLP(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        n = X.shape[0]
        self.net.train()
        iterator = range(self.epochs)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=desc, unit="epoch", leave=False)
        for epoch in iterator:
            perm = torch.randperm(n)
            last_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), y[idx])
                loss.backward()
                opt.step()
                last_loss = loss.item()
            if progress:
                iterator.set_postfix(loss=f"{last_loss:.4f}")
        return self

    def predict(self, X_raw):
        self.net.eval()
        Xs = self._scale(np.asarray(X_raw, dtype=np.float64))
        with torch.no_grad():
            out = self.net(torch.tensor(Xs, dtype=torch.float32))
        return out.numpy().astype(np.float64) * self._ysigma + self._ymu

    def fit_with_history(self, X_train_raw, y_train, X_eval_raw, y_eval, max_epochs=None):
        """Train like fit() but evaluate test RMSE after every epoch.

        Returns the per-epoch eval-RMSE list (length = epochs run). Used by the
        masking experiment to measure epochs-to-target. The fitted model is left
        ready for .predict(). `max_epochs` overrides self.epochs (e.g. an extended
        budget so a slow-to-converge masked model can still reach the target).
        """
        epochs = int(max_epochs or self.epochs)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        Xs = self._scale_fit(np.asarray(X_train_raw, dtype=np.float64))
        y_raw = np.asarray(y_train, dtype=np.float64)
        self._ymu = float(y_raw.mean())
        self._ysigma = float(y_raw.std()) or 1.0
        X = torch.tensor(Xs, dtype=torch.float32)
        y = torch.tensor((y_raw - self._ymu) / self._ysigma, dtype=torch.float32)

        self.net = _MLP(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        n = X.shape[0]
        y_eval = np.asarray(y_eval, dtype=np.float64)
        history = []
        for epoch in range(epochs):
            self.net.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), y[idx])
                loss.backward()
                opt.step()
            pred = self.predict(X_eval_raw)
            history.append(float(np.sqrt(np.mean((pred - y_eval) ** 2))))
        return history

    # --- persistence -------------------------------------------------------
    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "hidden": self.hidden,
            "mu": self._mu,
            "sigma": self._sigma,
            "ymu": self._ymu,
            "ysigma": self._ysigma,
            "input_dim": len(self._mu),
            "config": dict(lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                           weight_decay=self.weight_decay, seed=self.seed),
        }, path)
        return path

    @staticmethod
    def load(path):
        ckpt = torch.load(path, weights_only=False)
        m = NNModel(hidden=ckpt["hidden"], **ckpt["config"])
        m._mu = ckpt["mu"]
        m._sigma = ckpt["sigma"]
        m._ymu = ckpt.get("ymu", 0.0)
        m._ysigma = ckpt.get("ysigma", 1.0)
        m.net = _MLP(ckpt["input_dim"], ckpt["hidden"])
        m.net.load_state_dict(ckpt["state_dict"])
        m.net.eval()
        return m
