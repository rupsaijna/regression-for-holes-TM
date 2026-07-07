"""Error metric and distance / spacing utilities.

Distance statistics for large point sets are expensive (a full pairwise matrix
for the 16k-row train set would be ~2 GB).  So:
  * spread   -> mean distance to the centroid          : O(n), always exact
  * min dist -> nearest-neighbour distance (KD-tree)   : O(n log n), always exact
  * avg/max  -> exact via pairwise_distances when n is small, otherwise estimated
               from a fixed random sample of pairs (clearly reported as such)
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# Above this many points we stop materialising the full pairwise matrix and
# switch avg/max to sampling.
EXACT_PAIRWISE_MAX = 4000
SAMPLE_PAIRS = 2_000_000  # random pairs used when estimating avg/max
SAMPLE_SEED = 12345


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spread(X):
    """Mean Euclidean distance of points to their centroid."""
    c = X.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(X - c, axis=1).mean())


def min_pairwise_distance(X):
    """Exact smallest distance between any two distinct points (KD/ball tree)."""
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    dist, _ = nn.kneighbors(X)  # column 0 is the point itself (dist 0)
    return float(dist[:, 1].min())


def avg_max_pairwise_distance(X):
    """Return (avg, max, exact?) for pairwise distances.

    Exact for small sets; estimated from SAMPLE_PAIRS random pairs otherwise.
    """
    n = len(X)
    if n <= EXACT_PAIRWISE_MAX:
        D = pairwise_distances(X)
        iu = np.triu_indices(n, k=1)
        d = D[iu]
        return float(d.mean()), float(d.max()), True

    rng = np.random.default_rng(SAMPLE_SEED)
    i = rng.integers(0, n, size=SAMPLE_PAIRS)
    j = rng.integers(0, n, size=SAMPLE_PAIRS)
    mask = i != j
    i, j = i[mask], j[mask]
    d = np.linalg.norm(X[i] - X[j], axis=1)
    return float(d.mean()), float(d.max()), False


def distance_summary(X):
    """Convenience: dict of spread / avg / min / max for one point set."""
    avg, mx, exact = avg_max_pairwise_distance(X)
    return dict(
        n=len(X),
        spread=spread(X),
        avg_pairwise=avg,
        min_pairwise=min_pairwise_distance(X),
        max_pairwise=mx,
        avg_max_exact=exact,
    )


# --- per-point "spacing" definitions --------------------------------------
# Each returns one scalar per query point.

def spacing_nn_to_train(X_query, X_train, k=1):
    """Distance from each query point to its nearest training point(s) (mean of k)."""
    nn = NearestNeighbors(n_neighbors=k).fit(X_train)
    dist, _ = nn.kneighbors(X_query)
    return dist.mean(axis=1)


def spacing_local_density(X_query, X_all, k=5):
    """Mean distance to the k nearest points in the whole dataset (self excluded)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
    dist, _ = nn.kneighbors(X_query)
    return dist[:, 1:].mean(axis=1)  # drop self (distance 0)


def spacing_nn_to_self(X_query, k=1):
    """Distance from each query point to its nearest *other* query point (mean of k)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_query)
    dist, _ = nn.kneighbors(X_query)
    return dist[:, 1:1 + k].mean(axis=1)  # drop self


def all_spacings(X_test, X_train, X_all):
    """Return a dict {name: per-test-point spacing array} for every definition."""
    return {
        "nn_to_train_k1": spacing_nn_to_train(X_test, X_train, k=1),
        "knn_to_train_k5": spacing_nn_to_train(X_test, X_train, k=5),
        "local_density_k5": spacing_local_density(X_test, X_all, k=5),
        "nn_to_test_k1": spacing_nn_to_self(X_test, k=1),
    }


# --- per-point local-spacing-DISPERSION definitions ------------------------
# These measure how *uneven* a point's neighbourhood is (some neighbours close,
# some far) rather than how far neighbours are on average. They test the
# hypothesis "spacing variance, not spacing magnitude, drives error".

def _knn_train_dist(X_query, X_train, k=5):
    """Sorted ascending distances to the k nearest training points: (n_query, k)."""
    nn = NearestNeighbors(n_neighbors=k).fit(X_train)
    dist, _ = nn.kneighbors(X_query)
    return dist


def spacing_knn_cv(X_query, X_train, k=5):
    """Coefficient of variation (std/mean) of the k nearest-train distances.

    High = the k neighbours sit at very mixed distances (a density edge); low =
    a uniformly-spaced neighbourhood. Scale-free. NaN if the mean distance is 0.
    """
    d = _knn_train_dist(X_query, X_train, k)
    m = d.mean(axis=1)
    return d.std(axis=1) / np.where(m > 0, m, np.nan)


def spacing_knn_ratio(X_query, X_train, k=5):
    """Ratio d_k / d_1 (farthest / nearest of the k nearest-train distances).

    >= 1; large = the nearest neighbour is much closer than the k-th, i.e. the
    point sits on the edge of a cluster (very uneven local spacing). Scale-free.
    """
    d = _knn_train_dist(X_query, X_train, k)
    d1 = d[:, 0]
    return d[:, -1] / np.where(d1 > 0, d1, np.nan)


def spacing_knn_std(X_query, X_train, k=5):
    """Raw std of the k nearest-train distances (absolute, standardized units)."""
    return _knn_train_dist(X_query, X_train, k).std(axis=1)


def variance_spacings(X_test, X_train, k=5):
    """{name: per-test-point local-spacing-DISPERSION array} for every definition."""
    return {
        f"knn_cv_k{k}": spacing_knn_cv(X_test, X_train, k),
        f"knn_ratio_k{k}": spacing_knn_ratio(X_test, X_train, k),
        f"knn_std_k{k}": spacing_knn_std(X_test, X_train, k),
    }
