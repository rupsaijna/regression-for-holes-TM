"""Registry of regression datasets the experiments can run on.

Every dataset is chosen for being *well suited to linear regression* (an
approximately linear target-vs-feature relationship), drawn from:
  * sklearn builtin     - California Housing (the original baseline)
  * UCI ML Repository   - via the `ucimlrepo` package          (https://archive.ics.uci.edu)
  * Sutanoy/Public-Regression-Datasets - classic textbook sets (raw CSV/TXT on GitHub)

Each loader returns (X, Y, feature_names) as float64 numpy arrays and caches the
result to datasets/<key>.npz, so only the first run needs network access.

See DATASETS.md for the human-readable catalogue (links + descriptions).
"""
import io
import os
import urllib.request

import numpy as np

import config

DATASETS_DIR = os.path.join(config.BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

_SUTANOY = "https://raw.githubusercontent.com/Sutanoy/Public-Regression-Datasets/main/"

# NYSE pools ~500 symbols x ~1760 days (~850k rows); stride down to this many so
# the CPU Tsetlin Machine stays tractable (cf. metro ~48k -> ~2.5 min / run).
NYSE_MAX_ROWS = 50000


# --- caching ---------------------------------------------------------------
def _cache_path(key):
    return os.path.join(DATASETS_DIR, f"{key}.npz")


def _cached(key, builder):
    """Return (X, Y, feature_names), building + caching on first use."""
    path = _cache_path(key)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"].astype(np.float64), d["Y"].astype(np.float64), list(d["feat"])
    X, Y, feat = builder()
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).ravel()
    np.savez(path, X=X, Y=Y, feat=np.array(feat, dtype=object))
    return X, Y, feat


def _http_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


# --- individual loaders ----------------------------------------------------
def _load_california():
    from sklearn.datasets import fetch_california_housing
    ds = fetch_california_housing()
    return ds.data, ds.target, list(ds.feature_names)


def _uci_loader(uci_id, target_col=None, na_values=None, max_missing=1.0, row_filter=None):
    """Build a loader for a UCI dataset id (numeric features only).

    target_col   : target name; searched in features first, then targets; if None
                   the first target column is used.
    na_values    : values to treat as missing (e.g. [-200] for Air Quality).
    max_missing  : drop feature columns whose missing fraction exceeds this
                   (e.g. 0.5 to drop the mostly-empty NMHC column).
    row_filter   : optional callable(df)->bool mask to drop bad rows (sentinels).
    """
    def build():
        import pandas as pd
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=uci_id)
        feats = ds.data.features.copy()
        tgts = ds.data.targets
        if target_col and target_col in feats.columns:
            y = feats[target_col]; feats = feats.drop(columns=[target_col])
        elif target_col and tgts is not None and target_col in tgts.columns:
            y = tgts[target_col]
        elif tgts is not None and tgts.shape[1] >= 1:
            y = tgts.iloc[:, 0]
        else:
            raise ValueError(f"UCI id={uci_id}: no usable target column")

        df = feats.select_dtypes(include=[np.number]).copy()
        df["__y__"] = pd.to_numeric(y, errors="coerce").to_numpy()
        if na_values:
            df = df.replace(list(na_values), np.nan)
        df = df.loc[:, [c for c in df.columns if c == "__y__" or df[c].isna().mean() <= max_missing]]
        if row_filter is not None:
            df = df[row_filter(df)]
        df = df.dropna()
        feat = [c for c in df.columns if c != "__y__"]
        return df[feat].to_numpy(), df["__y__"].to_numpy(), feat
    return build


def _local_csv_loader(filename, target_col=None, drop_cols=(), sep=","):
    """Build a loader for a user-provided CSV in datasets/ (e.g. Kaggle data).

    target_col : target column name; if None the last numeric column is used.
    Raises a clear error (with instructions) if the file is missing.
    """
    def build():
        import pandas as pd
        path = os.path.join(DATASETS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected '{path}' but it is not there.\n"
                f"    This dataset is not downloadable headlessly (e.g. a Kaggle\n"
                f"    competition). Place the CSV at that path, then re-run. See DATASETS.md.")
        df = pd.read_csv(path, sep=sep)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.select_dtypes(include=[np.number]).dropna()
        tgt = target_col if target_col else df.columns[-1]
        feat = [c for c in df.columns if c != tgt]
        return df[feat].to_numpy(), df[tgt].to_numpy(), feat
    return build


def _sutanoy_loader(filename, target_col, drop_cols=(), sep=r"\s+"):
    """Build a loader for a whitespace/CSV file in the Sutanoy repo (numeric only)."""
    def build():
        import pandas as pd
        df = pd.read_csv(io.StringIO(_http_text(_SUTANOY + filename)), sep=sep)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.select_dtypes(include=[np.number]).dropna()
        feat = [c for c in df.columns if c != target_col]
        return df[feat].to_numpy(), df[target_col].to_numpy(), feat
    return build


def _load_nyse_nextvol(log_target=False):
    """Next-day trading-volume regression built from the NYSE price history.

    Source: datasets/nyse/prices-split-adjusted.csv (Kaggle `dgawlik/nyse`) -- the
    daily OHLCV bars of ~500 S&P-500 names, 2010-2016. We turn the per-symbol
    time series into a supervised table: each row is one trading day for one
    symbol, the features describe THAT day (plus two short volume lags), and the
    target is the SAME symbol's volume on the NEXT trading day.

    Symbol identity is dropped (numeric features only, like the rest of the
    pipeline); today's own volume and the 5-day average volume carry the
    per-symbol scale. Pooled across symbols the table is ~850k rows, so it is
    deterministically strided down to NYSE_MAX_ROWS to stay tractable for the
    CPU Tsetlin Machine.

    If `log_target` is True the target is natural-log volume instead of raw
    volume (features are identical) -- a much less heavy-tailed target, used to
    test whether the spacing->error correlation on raw `nyse` is intrinsic or a
    heavy-tail artefact.
    """
    def build():
        import pandas as pd
        path = os.path.join(DATASETS_DIR, "nyse", "prices-split-adjusted.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected '{path}'.\n"
                f"    Place the Kaggle NYSE dataset (dgawlik/nyse) under datasets/nyse/. "
                f"See DATASETS.md.")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        g = df.groupby("symbol", sort=False)
        # today's-bar engineered features (all known at prediction time)
        df["ret"] = (df["close"] - df["open"]) / df["open"]          # intraday return
        df["rng"] = (df["high"] - df["low"]) / df["open"]            # intraday range
        df["vol_lag1"] = g["volume"].shift(1)                        # yesterday's volume
        df["vol_ma5"] = g["volume"].transform(                      # 5-day avg incl. today
            lambda s: s.rolling(5, min_periods=5).mean())
        df["y"] = g["volume"].shift(-1)                              # <- target: next-day volume
        feat = ["open", "high", "low", "close", "volume", "ret", "rng", "vol_lag1", "vol_ma5"]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat + ["y"])
        df = df[(df["volume"] > 0) & (df["y"] > 0)]                  # drop halted / zero-volume days
        if len(df) > NYSE_MAX_ROWS:
            df = df.iloc[:: len(df) // NYSE_MAX_ROWS]                # deterministic stride
        y = df["y"].to_numpy()
        if log_target:
            y = np.log(y)                                           # volume > 0 guaranteed above
        return df[feat].to_numpy(), y, feat
    return build


# --- registry --------------------------------------------------------------
# key -> dict(meta + builder). `builder` returns (X, Y, feature_names).
DATASETS = {
    "california": dict(
        title="California Housing", source="sklearn", url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
        n=20640, d=8, target="median house value ($100k)",
        linear="Moderate (linear R^2 ~ 0.6); the original baseline dataset.",
        desc="Median house value per California block group from the 1990 census, with 8 numeric predictors (income, rooms, location, ...).",
        builder=_load_california),
    "ccpp": dict(
        title="Combined Cycle Power Plant", source="UCI", url="https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
        n=9568, d=4, target="net hourly electrical energy output PE (MW)",
        linear="Excellent (linear R^2 ~ 0.93). The cleanest large linear set here.",
        desc="Ambient conditions (temperature, pressure, humidity, exhaust vacuum) vs the net electrical output of a combined-cycle power plant.",
        builder=_uci_loader(294)),
    "energy": dict(
        title="Energy Efficiency", source="UCI", url="https://archive.ics.uci.edu/dataset/242/energy+efficiency",
        n=768, d=8, target="heating load Y1",
        linear="Good (heating load is close to linear in the building parameters).",
        desc="Simulated heating/cooling loads of buildings from 8 shape/glazing parameters; target here is the heating load (Y1).",
        builder=_uci_loader(242, target_col="Y1")),
    "autompg": dict(
        title="Auto MPG", source="UCI", url="https://archive.ics.uci.edu/dataset/9/auto+mpg",
        n=392, d=7, target="fuel efficiency (mpg)",
        linear="Good (linear R^2 ~ 0.8). 6 rows with missing horsepower are dropped.",
        desc="Fuel consumption of late-1970s/early-1980s cars vs cylinders, displacement, horsepower, weight, acceleration, model year, origin.",
        builder=_uci_loader(9)),
    "realestate": dict(
        title="Real Estate Valuation (Taipei)", source="UCI", url="https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set",
        n=414, d=6, target="house price of unit area",
        linear="Moderate-good; a standard small multiple-regression set.",
        desc="Taipei housing: price per unit area vs transaction date, house age, distance to MRT, number of nearby convenience stores, lat/long.",
        builder=_uci_loader(477)),
    "mortality": dict(
        title="Mortality vs Weather & Pollution", source="Sutanoy", url="https://github.com/Sutanoy/Public-Regression-Datasets/blob/main/mortality_weather.txt",
        n=60, d=15, target="age-adjusted mortality rate (Death)",
        linear="Classic multiple-linear-regression set (McDonald & Schwing pollution data).",
        desc="60 US metro areas: age-adjusted mortality regressed on 15 climate, socioeconomic and air-pollution variables. Small but a textbook linear case.",
        builder=_sutanoy_loader("mortality_weather.txt", target_col="Death", drop_cols=["Ind"])),
    "bloodfat": dict(
        title="Blood Fat (Age & Weight)", source="Sutanoy", url="https://github.com/Sutanoy/Public-Regression-Datasets/blob/main/age_wt_bfat.txt",
        n=25, d=2, target="blood fat concentration",
        linear="Textbook simple/multiple linear regression (very linear, tiny).",
        desc="Blood fat concentration of 25 people vs body weight and age; a minimal, strongly linear example for illustration.",
        builder=_sutanoy_loader("age_wt_bfat.txt", target_col="Blood_fat", drop_cols=["Index"])),
    # --- UCI time-series regression (added for UNEVEN spacing) -------------
    "metro": dict(
        title="Metro Interstate Traffic Volume", source="UCI (time-series)", url="https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        n=48204, d=4, target="hourly traffic volume",
        linear="Weakly linear, but selected for EXTREMELY uneven spacing (norm-entropy ~0.00): "
               "strongly bimodal day/night structure. Temp=0 K and a rain outlier are filtered.",
        desc="Hourly westbound I-94 traffic volume vs weather (temperature, rain, snow, cloud cover). "
             "The most unevenly-spaced dataset in the registry.",
        builder=_uci_loader(492, target_col="traffic_volume",
                            row_filter=lambda df: (df["temp"] > 0) & (df["rain_1h"] < 1000))),
    "beijing": dict(
        title="Beijing PM2.5", source="UCI (time-series)", url="https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data",
        n=41757, d=10, target="PM2.5 concentration (ug/m^3)",
        linear="Moderately linear; selected for genuinely uneven spacing (skew ~4.5, norm-entropy ~0.39).",
        desc="Hourly PM2.5 in Beijing vs dew point, temperature, pressure, wind speed, hours of snow/rain "
             "and the date/time index. Rows with missing PM2.5 are dropped.",
        builder=_uci_loader(381, target_col="pm2.5")),
    "airquality": dict(
        title="Air Quality (UCI)", source="UCI (time-series)", url="https://archive.ics.uci.edu/dataset/360/air+quality",
        n=6941, d=11, target="benzene concentration C6H6(GT)",
        linear="Reasonably linear among the sensor channels; moderately uneven spacing (norm-entropy ~0.60).",
        desc="Hourly responses of 5 metal-oxide gas sensors plus temperature/humidity in an Italian city, "
             "predicting true benzene concentration. -200 missing markers and the mostly-empty NMHC column are removed.",
        builder=_uci_loader(360, target_col="C6H6(GT)", na_values=[-200], max_missing=0.5)),
    # --- Kaggle (user-provided local CSV; not downloadable headlessly) -----
    "stock": dict(
        title="Kaggle Stock Market Prediction", source="Kaggle", url="https://www.kaggle.com/competitions/kaggle-stock-market-prediction/data",
        n=None, d=None, target="(last numeric column, or set in registry)",
        linear="TBD -- depends on the chosen target/features once the data is provided.",
        desc="Kaggle competition stock data. Competition files need Kaggle auth + rule acceptance, so they "
             "cannot be fetched headlessly: place the CSV at datasets/kaggle_stock.csv (see DATASETS.md).",
        builder=_local_csv_loader("kaggle_stock.csv")),
    "nyse": dict(
        title="NYSE Next-Day Volume", source="Kaggle (NYSE)", url="https://www.kaggle.com/datasets/dgawlik/nyse",
        n=NYSE_MAX_ROWS, d=9, target="next trading day's volume (shares)",
        linear="Moderate -- volume is strongly autocorrelated, so today's volume and the 5-day "
               "average are good linear predictors; the target is heavy-tailed.",
        desc="Daily split-adjusted OHLCV bars of ~500 S&P-500 names (2010-2016). Each row predicts the "
             "SAME symbol's NEXT-day volume from today's open/high/low/close/volume, intraday return & range, "
             "yesterday's volume and the 5-day average volume. Pooled across symbols and strided to ~50k rows.",
        builder=_load_nyse_nextvol()),
    "nyse_log": dict(
        title="NYSE Next-Day Volume (log)", source="Kaggle (NYSE)", url="https://www.kaggle.com/datasets/dgawlik/nyse",
        n=NYSE_MAX_ROWS, d=9, target="log(next trading day's volume)",
        linear="Same features as `nyse`; log target is far less heavy-tailed. Used to test whether the "
               "spacing->error correlation on raw nyse is intrinsic or a heavy-tail artefact.",
        desc="Identical to `nyse` (same OHLCV + lag features, same rows) but the target is natural-log "
             "next-day volume instead of raw volume.",
        builder=_load_nyse_nextvol(log_target=True)),
}

DEFAULT_DATASET = "california"


def load(key):
    """Return (X, Y, feature_names) for a registered dataset key (cached)."""
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{key}'. Available: {', '.join(DATASETS)}")
    return _cached(key, DATASETS[key]["builder"])


def describe_table():
    """One-line-per-dataset summary string (used by --list-datasets)."""
    rows = [f"{'key':12s} {'source':18s} {'n':>6s} {'d':>3s}  title / linear-fit"]
    for k, m in DATASETS.items():
        n = "?" if m["n"] is None else str(m["n"])
        d = "?" if m["d"] is None else str(m["d"])
        rows.append(f"{k:12s} {m['source']:18s} {n:>6s} {d:>3s}  {m['title']} -- {m['linear']}")
    return "\n".join(rows)
