"""Training-set masking utilities for the Phase-7 masking experiments.

Two ways to remove a fraction `p` of the training points:
  * uniform    -- drop p.n points uniformly at random (sparser, still even).
  * structured -- drop p.n points by carving HOLES (random seed points + their
                  nearest neighbours), which makes the surviving set genuinely
                  uneven (dense regions + empty regions).

Plus a region-split helper that labels each *test* point by how much of its local
training support was removed -- the "masked test set" (in-hole vs surviving).

All geometry is computed in the standardized feature space the rest of the
pipeline uses, so distances are comparable across features.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def mask_uniform(n, p, seed):
    """Drop p.n of range(n) uniformly at random. Returns (kept_idx, removed_idx)."""
    n_remove = int(round(p * n))
    rng = np.random.default_rng(seed)
    removed = rng.choice(n, size=n_remove, replace=False) if n_remove else np.array([], dtype=int)
    kept = np.setdiff1d(np.arange(n), removed, assume_unique=False)
    return np.sort(kept), np.sort(removed)


def mask_structured(X_std, p, seed, hole_frac=0.05):
    """Drop ~p.n points by carving holes: pick a random surviving seed, remove it
    and its nearest neighbours; repeat until p.n removed. Returns (kept_idx, removed_idx).

    hole_frac sets the hole size (fraction of n per hole); the number of holes is
    therefore ~p/hole_frac. Deterministic given `seed`.
    """
    n = len(X_std)
    target = int(round(p * n))
    if target <= 0:
        return np.arange(n), np.array([], dtype=int)
    hole_size = max(1, int(round(hole_frac * n)))
    nn = NearestNeighbors(n_neighbors=min(hole_size, n)).fit(X_std)
    rng = np.random.default_rng(seed)
    removed = set()
    while len(removed) < target:
        survivors = np.fromiter((i for i in range(n) if i not in removed), dtype=int)
        seed_pt = int(rng.choice(survivors))
        order = nn.kneighbors(X_std[seed_pt:seed_pt + 1], return_distance=False)[0]
        for j in order:
            if len(removed) >= target:
                break
            removed.add(int(j))
    removed_idx = np.sort(np.fromiter(removed, dtype=int))
    kept = np.setdiff1d(np.arange(n), removed_idx, assume_unique=True)
    return kept, removed_idx


def region_split(X_std_train, removed_idx, X_std_test, k=5):
    """Label each test point in-hole if >= half its k nearest ORIGINAL-train
    neighbours were removed. Returns a boolean mask over the test points (True = in-hole).
    """
    if len(removed_idx) == 0:
        return np.zeros(len(X_std_test), dtype=bool)
    nn = NearestNeighbors(n_neighbors=min(k, len(X_std_train))).fit(X_std_train)
    neigh = nn.kneighbors(X_std_test, return_distance=False)  # (n_test, k) train indices
    removed_mask = np.zeros(len(X_std_train), dtype=bool)
    removed_mask[removed_idx] = True
    frac_removed = removed_mask[neigh].mean(axis=1)
    return frac_removed >= 0.5
