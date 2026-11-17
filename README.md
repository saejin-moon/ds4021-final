# DS 4021 — Kepler Objects of Interest (KOI) Classification Project

## Repository Structure

```text
├── Data/
│   ├── README.md            # Guidelines keeping the 20% reporting set untouched
│   └── reporting_set.csv    # Untouched 20% stratified reporting split (1,320 rows)
├── data_working/            # Ignored by Git; stores local 80% working split (5,278 rows)
├── docs/                    # Project proposal PDF and documentation
├── src/
│   └── prepare_data.py      # Reproducible cleaning, summary stats, and 80/20 split script
├── koi.csv                  # Raw NASA Exoplanet Archive KOI cumulative table
└── .gitignore
```

## Dataset Source

- **Source:** NASA Exoplanet Archive — Kepler Objects of Interest (KOI) Cumulative Table (`http://exoplanetarchive.ipac.caltech.edu`, exported `2026-11-17`).
- **Target Variable:** `koi_disposition` (`FALSE POSITIVE`, `CONFIRMED`, `CANDIDATE`).

## Reproducing the Data Split

Run `src/prepare_data.py` with an environment containing `pandas` and `scikit-learn`:

```bash
python src/prepare_data.py
```

The script:
1. Loads `koi.csv` (`comment="#"`).
2. Drops non-predictive ID and text metadata columns (`rowid`, `kepid`, and all non-numeric metadata columns except `koi_disposition`), empty (100% missing) columns, and zero-variance columns.
3. Drops rows containing missing values, producing a completely dense dataset of **N = 6,598** observations and **P = 100** numeric predictor features.
4. Performs a stratified 80/20 split (`test_size=0.20`, `random_state=42`, stratified on `koi_disposition`), saving `Data/reporting_set.csv` (1,320 rows) and `data_working/working_set.csv` (5,278 rows).
5. Prints full-dataset summary statistics (sample size, feature count, target class counts/percentages, and mean, SD, median, and IQR for representative features).

## Dataset Summary (Full Dense Dataset, N = 6,598, P = 100)

| `koi_disposition` | Count | Percentage |
| --- | ---: | ---: |
| `FALSE POSITIVE` | 2,879 | 43.63% |
| `CONFIRMED` | 2,479 | 37.57% |
| `CANDIDATE` | 1,240 | 18.79% |
