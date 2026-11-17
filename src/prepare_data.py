"""Prepare the Kepler Objects of Interest (KOI) dataset for DS 4021.

Loads the raw NASA Exoplanet Archive KOI CSV, removes non-predictive ID/text
metadata, empty, and zero-variance columns, drops rows with missing values to
produce a dense dataset ready for scikit-learn/PyTorch pipelines, performs a
stratified 80/20 split (random_state=42), saves the splits, and prints full-
dataset summary statistics for the project proposal rubric.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "koi_disposition"
NON_PREDICTIVE_ID_COLS = ("rowid", "kepid")
# Robovetter vetting outputs directly encode the disposition label (target leakage).
VETTING_LEAKAGE_COLS = (
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
)
EXCLUDED_NUMERIC_COLS = NON_PREDICTIVE_ID_COLS + VETTING_LEAKAGE_COLS
REPRESENTATIVE_FEATURES = (
    ("koi_period", "Orbital Period [days]"),
    ("koi_duration", "Transit Duration [hrs]"),
    ("koi_depth", "Transit Depth [ppm]"),
    ("koi_prad", "Planetary Radius [Earth radii]"),
    ("koi_steff", "Stellar Effective Temp [K]"),
)
TEST_SIZE = 0.20
RANDOM_STATE = 42


def prepare_dataset(raw_csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and clean the raw KOI CSV into a dense modeling DataFrame."""
    raw_df = pd.read_csv(raw_csv_path, comment="#")

    if TARGET_COL not in raw_df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in {raw_csv_path}")

    # Drop completely empty columns before inspecting numeric predictors.
    non_empty_df = raw_df.dropna(axis=1, how="all")

    numeric_cols = non_empty_df.select_dtypes(include="number").columns
    feature_cols = [
        col
        for col in numeric_cols
        if col not in EXCLUDED_NUMERIC_COLS and non_empty_df[col].nunique(dropna=True) > 1
    ]

    dense_df = non_empty_df[feature_cols + [TARGET_COL]].dropna().reset_index(drop=True)
    return dense_df, feature_cols


def print_summary(dense_df: pd.DataFrame, feature_cols: list[str], working_df: pd.DataFrame, reporting_df: pd.DataFrame) -> None:
    """Print dataset dimensions, class distribution, and representative feature stats."""
    n_samples = len(dense_df)
    n_features = len(feature_cols)

    print("=" * 72)
    print("DS 4021 — KOI Dataset Preparation Summary")
    print("=" * 72)
    print(f"Final dense sample size (N): {n_samples}")
    print(f"Number of predictor features (P): {n_features}")
    print(f"Missing values remaining: {int(dense_df.isna().sum().sum())}")
    print(f"80% working set size:   {len(working_df)} rows ({len(working_df) / n_samples:.1%})")
    print(f"20% reporting set size: {len(reporting_df)} rows ({len(reporting_df) / n_samples:.1%})")

    print("\nTarget Variable Distribution (koi_disposition):")
    counts = dense_df[TARGET_COL].value_counts()
    for label, count in counts.items():
        pct = 100.0 * count / n_samples
        print(f"  {label:<16s} {count:5d} ({pct:5.2f}%)")

    print("\nRepresentative Feature Summary Statistics (Full Dense Dataset):")
    header = f"  {'Feature':<14s} {'Description':<32s} {'Mean':>12s} {'SD':>12s} {'Median':>12s} {'IQR':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for col, desc in REPRESENTATIVE_FEATURES:
        series = dense_df[col]
        mean = series.mean()
        std = series.std()
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        print(f"  {col:<14s} {desc:<32s} {mean:12.4f} {std:12.4f} {median:12.4f} {iqr:12.4f}")
    print("=" * 72)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_csv_path = repo_root / "koi.csv"
    reporting_path = repo_root / "Data" / "reporting_set.csv"
    working_path = repo_root / "data_working" / "working_set.csv"

    reporting_path.parent.mkdir(parents=True, exist_ok=True)
    working_path.parent.mkdir(parents=True, exist_ok=True)

    dense_df, feature_cols = prepare_dataset(raw_csv_path)

    working_df, reporting_df = train_test_split(
        dense_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dense_df[TARGET_COL],
    )

    reporting_df.to_csv(reporting_path, index=False)
    working_df.to_csv(working_path, index=False)

    print_summary(dense_df, feature_cols, working_df, reporting_df)
    print(f"Saved 20% reporting set -> {reporting_path.relative_to(repo_root)}")
    print(f"Saved 80% working set   -> {working_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
