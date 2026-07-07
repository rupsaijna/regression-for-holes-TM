# Regression Datasets

The baseline (`experiment1_baseline.py`) and analytics (`experiment2_analytics.py`)
programs can run on any dataset below via `--dataset <key>`. List them at runtime
with `python experiment1_baseline.py --list-datasets`.

## Selection criteria
Two families of datasets live here:

1. **Linear-fit sets** — chosen for an approximately linear target-vs-predictor
   relationship, so a linear model already does well (the original comparison goal).
2. **Uneven-spacing time-series sets** — UCI *time-series* regression datasets added
   specifically because their test points are **unevenly spaced** in feature space
   (the property the spacing-vs-error analysis probes). They were ranked by the
   `norm_entropy` of the test-point spacing distribution (see `spacing_distribution.py`):
   lower entropy = spacing concentrated/bimodal = more uneven. `metro` is the most
   uneven in the whole registry.

Sources:

- **sklearn builtin** — California Housing (the starting point)
- **UCI ML Repository** — https://archive.ics.uci.edu (fetched via the `ucimlrepo` package),
  including the [time-series regression](https://archive.ics.uci.edu/datasets?Task=Regression&Types=Time-Series) subset
- **Sutanoy/Public-Regression-Datasets** — https://github.com/Sutanoy/Public-Regression-Datasets (classic textbook sets)
- **Kaggle** — competition data that must be downloaded with Kaggle auth + rule
  acceptance, so it cannot be fetched headlessly (see `stock` below)

Each loader keeps **numeric features only**, drops rows with missing values, and
caches the result to `datasets/<key>.npz` (so only the first run needs network).

## Catalogue

| key | dataset | source | n | features | target | linear-fit suitability |
|-----|---------|--------|---|----------|--------|------------------------|
| `california` | California Housing | sklearn | 20640 | 8 | median house value ($100k) | Moderate (R²≈0.6) — baseline |
| `ccpp` | Combined Cycle Power Plant | UCI | 9568 | 4 | net electrical output PE (MW) | **Excellent (R²≈0.93)** |
| `energy` | Energy Efficiency | UCI | 768 | 8 | heating load (Y1) | Good |
| `autompg` | Auto MPG | UCI | 392 | 7 | fuel efficiency (mpg) | Good (R²≈0.8) |
| `realestate` | Real Estate Valuation (Taipei) | UCI | 414 | 6 | price per unit area | Moderate–good |
| `mortality` | Mortality vs Weather & Pollution | Sutanoy | 60 | 15 | age-adjusted mortality | Classic linear (small) |
| `bloodfat` | Blood Fat (Age & Weight) | Sutanoy | 25 | 2 | blood fat concentration | Textbook linear (tiny) |
| `metro` | Metro Interstate Traffic Volume | UCI (time-series) | 48204 | 4 | hourly traffic volume | Weak linear — **most uneven spacing** (norm-entropy ≈ 0.00) |
| `beijing` | Beijing PM2.5 | UCI (time-series) | 41757 | 10 | PM2.5 (µg/m³) | Moderate linear; uneven spacing (skew ≈ 4.5, norm-entropy ≈ 0.39) |
| `airquality` | Air Quality (UCI) | UCI (time-series) | 6941 | 11 | benzene C6H6(GT) | Reasonably linear; moderately uneven (norm-entropy ≈ 0.60) |
| `stock` | Kaggle Stock Market Prediction | Kaggle | — | — | (last numeric col) | TBD — needs local CSV (see below) |
| `nyse` | NYSE Next-Day Volume | Kaggle (NYSE) | ~50000 | 9 | next trading day's volume (shares) | Moderate — autocorrelated volume; heavy-tailed target |
| `nyse_log` | NYSE Next-Day Volume (log) | Kaggle (NYSE) | ~50000 | 9 | log(next-day volume) | Same features as `nyse`; well-behaved target (heavy-tail control) |

### california — California Housing
Median house value per California block group (1990 census) against 8 numeric
predictors (median income, average rooms, house age, location, …). The original
baseline; linear regression reaches R²≈0.6.
🔗 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### ccpp — Combined Cycle Power Plant
6 years of hourly readings: ambient **temperature, atmospheric pressure, relative
humidity and exhaust vacuum** vs the plant's **net electrical energy output (PE)**.
The relationship is almost perfectly linear (linear-regression R²≈0.93), making it
the cleanest large linear benchmark in this set.
🔗 https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

### energy — Energy Efficiency
768 simulated building shapes described by 8 parameters (relative compactness,
surface/wall/roof area, height, orientation, glazing). Two possible targets —
**heating load (Y1)** is used here; it is close to linear in the inputs.
🔗 https://archive.ics.uci.edu/dataset/242/energy+efficiency

### autompg — Auto MPG
Fuel consumption (**mpg**) of late-1970s/early-1980s cars vs cylinders,
displacement, horsepower, weight, acceleration, model year and origin. The 6 rows
with missing horsepower are dropped (→ 392 rows). Linear-regression R²≈0.8.
🔗 https://archive.ics.uci.edu/dataset/9/auto+mpg

### realestate — Real Estate Valuation (Taipei)
414 Taipei transactions: **price per unit area** vs transaction date, house age,
distance to the nearest MRT station, number of nearby convenience stores, and
latitude/longitude. A standard small multiple-regression dataset.
🔗 https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set

### mortality — Mortality vs Weather & Pollution
The classic **McDonald & Schwing** pollution dataset: 60 US metropolitan areas,
**age-adjusted mortality rate** regressed on 15 climate, socioeconomic and
air-pollution variables. Small but a textbook multiple-linear-regression case.
🔗 https://github.com/Sutanoy/Public-Regression-Datasets/blob/main/mortality_weather.txt

### bloodfat — Blood Fat (Age & Weight)
A minimal teaching dataset: **blood fat concentration** of 25 people vs body
weight and age. Strongly linear; included as the simplest possible illustration.
🔗 https://github.com/Sutanoy/Public-Regression-Datasets/blob/main/age_wt_bfat.txt

## Uneven-spacing time-series datasets

These three were added from the UCI **time-series regression** subset specifically
for their **uneven test-point spacing**, which is what the spacing-vs-error analysis
(`experiment2_analytics.py`, `correlation_summary.py`) is designed to probe. Linear
suitability was a secondary concern here.

### metro — Metro Interstate Traffic Volume
Hourly westbound I-94 traffic volume vs weather (**temperature, rain, snow, cloud
cover**). The strong day/night bimodality makes this the **most unevenly-spaced
dataset in the registry** (norm-entropy ≈ 0.00). Rows with `temp = 0 K` and a rain
sensor outlier (`rain_1h ≥ 1000`) are filtered out. Only weakly linear — kept for the
spacing geometry, not the linear fit.
🔗 https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

### beijing — Beijing PM2.5
Hourly **PM2.5 concentration** in Beijing vs dew point, temperature, pressure, wind
speed, cumulative snow/rain hours and the date/time index. Genuinely uneven spacing
(skew ≈ 4.5, norm-entropy ≈ 0.39) with a long tail of isolated high-pollution points.
Rows with missing PM2.5 are dropped.
🔗 https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data

### airquality — Air Quality (UCI)
Hourly responses of **5 metal-oxide gas sensors** plus temperature/humidity in an
Italian city, predicting true **benzene concentration C6H6(GT)**. Moderately uneven
spacing (norm-entropy ≈ 0.60) and reasonably linear among the sensor channels. The
`-200` missing markers and the mostly-empty NMHC column are removed.
🔗 https://archive.ics.uci.edu/dataset/360/air+quality

## Kaggle dataset (local CSV required)

### stock — Kaggle Stock Market Prediction
Kaggle competition stock data. Competition files require Kaggle authentication **and**
acceptance of the competition rules, so they **cannot be fetched headlessly**. To use
this dataset:

1. Download the data from
   https://www.kaggle.com/competitions/kaggle-stock-market-prediction/data
2. Place the CSV at **`datasets/kaggle_stock.csv`**
3. Run `python experiment1_baseline.py --dataset stock` (the loader keeps numeric
   columns and uses the last one as the target unless the registry sets `target_col`).

Until the CSV is present, any run on `stock` raises a clear `FileNotFoundError` with
these instructions.
🔗 https://www.kaggle.com/competitions/kaggle-stock-market-prediction/data

### nyse — NYSE Next-Day Volume
A **next-day trading-volume** prediction task built from the Kaggle `dgawlik/nyse`
dataset (daily split-adjusted OHLCV bars of ~500 S&P-500 names, 2010–2016), placed
under `datasets/nyse/`. The per-symbol time series is turned into a supervised table:

- **One row = one trading day for one symbol.**
- **Features (9, all known at the close of "today"):** `open`, `high`, `low`,
  `close`, `volume`, intraday return `ret = (close−open)/open`, intraday range
  `rng = (high−low)/open`, yesterday's volume `vol_lag1`, and the 5-day average
  volume `vol_ma5`.
- **Target:** the **same symbol's volume on the next trading day**
  (`volume` shifted −1 within the symbol).

Symbol identity is dropped (the pipeline uses numeric features only); today's own
volume and the 5-day average carry the per-symbol scale. Halted / zero-volume days
(feature or target `volume ≤ 0`) are removed. Pooled across all symbols the table is
~850k rows, so it is **deterministically strided down to ~50k rows**
(`NYSE_MAX_ROWS` in `common/datasets.py`) to stay tractable for the CPU Tsetlin
Machine. The source file used is `prices-split-adjusted.csv`.

> **Note — heavy-tailed target.** Daily volume ranges from ~10² to ~6×10⁸ shares
> (median ≈ 2.5M, max ≈ 655M). Predicting **raw** next-day volume (as the objective
> asks) is therefore a hard, long-tailed regression; predicting **log-volume** is the
> natural, much better-behaved variant if the raw-volume fit proves too coarse.

🔗 https://www.kaggle.com/datasets/dgawlik/nyse

## A note on the small datasets
`mortality` (60 rows) and `bloodfat` (25 rows) are genuine textbook *linear*
datasets, but they are **too small for the Tsetlin Machine** to shine: the TM
needs many samples and its feature binarization is near-degenerate on a handful of
rows, so the MLP (with target standardization) clearly wins there. They are kept
because they are canonical linear-regression examples and useful for sanity
checks; the larger sets (`ccpp`, `california`, `energy`) are where the
TM-vs-NN-vs-spacing comparison is most meaningful.

## Usage

```bash
# list everything
python experiment1_baseline.py --list-datasets

# baseline on a UCI set (cumulative avg RMSE at 10/50/100 runs, save models)
python experiment1_baseline.py --dataset ccpp --runs 10 50 100 --save-model

# analytics on the same dataset (uses the saved models + that dataset's fixed split)
python experiment2_analytics.py --dataset ccpp --slice 0:50 --buckets 8
```

Each dataset has its **own** fixed train/test split (`splits/<key>.npz`) and its own
canonical models (`saved_models/<key>__tm.pkl`, `saved_models/<key>__nn.pt`) and
analytics figures (`outputs/analytics/<key>/`), so datasets never overwrite each
other.
