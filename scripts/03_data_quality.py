# =============================================================================
# scripts/03_data_quality.py
# -----------------------------------------------------------------------------
# Stage 03: Data Quality Assessment
#
# What it does:
#   - Loads all CSVs from the dataset folder
#   - Computes per-feature quality metrics:
#       missing_pct, n_unique, min, max, mean, std, n_outliers (IQR method)
#   - Saves a full quality report CSV
#   - Saves a missing data bar chart (SVG)
#   - Saves an outlier count bar chart (SVG)
#
# Outputs (in outputs/<DATASET_NAME>/):
#   03_data_quality_report.csv
#   03_missing_data_barchart.svg
#   03_outlier_counts.svg
#
# Log file:
#   logs/03_data_quality_<DATASET_NAME>.txt
#
# Usage:
#   python scripts/03_data_quality.py
#   Change DATASET_NAME below to switch datasets.
# =============================================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from utils.data_loader import get_dataset_path, load_csvs
from utils.logger import get_logger
from utils.plotting import save_figure, set_thesis_style

# ---------------------------------------------------------------------------
# Configuration — change this to switch datasets
# ---------------------------------------------------------------------------
DATASET_NAME = "tihm"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logger     = get_logger("03_data_quality", DATASET_NAME)
OUTPUT_DIR = config.OUTPUTS_DIR / DATASET_NAME


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def count_iqr_outliers(series):
    """Count values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] for a numeric series."""
    series = series.dropna()
    if len(series) == 0:
        return 0
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def compute_quality_metrics(df, filename):
    """Compute quality metrics for every column in *df*.

    Returns a list of dicts, one per column.
    """
    records = []
    n = len(df)

    for col in df.columns:
        series     = df[col]
        missing    = series.isnull().sum()
        is_numeric = pd.api.types.is_numeric_dtype(series)

        record = {
            "feature":      col,
            "source_file":  filename,
            "missing_pct":  round(missing / n * 100, 2) if n > 0 else np.nan,
            "n_unique":     series.nunique(),
            "min":          series.min()  if is_numeric else np.nan,
            "max":          series.max()  if is_numeric else np.nan,
            "mean":         round(series.mean(), 4) if is_numeric else np.nan,
            "std":          round(series.std(),  4) if is_numeric else np.nan,
            "n_outliers":   count_iqr_outliers(series) if is_numeric else np.nan,
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_thesis_style()

    # --- 1. Load data ---
    data_path = get_dataset_path(DATASET_NAME)
    logger.info("Dataset path: %s", data_path)

    dataframes = load_csvs(data_path)

    if not dataframes:
        logger.warning("No data found. Run the appropriate downloader first.")
        return

    # --- 2. Compute quality metrics ---
    all_records = []
    for filename, df in dataframes.items():
        logger.info("Computing quality metrics for: %s (%d rows)", filename, len(df))
        all_records.extend(compute_quality_metrics(df, filename))

    quality_df = pd.DataFrame(all_records)
    logger.info("Quality report: %d features assessed", len(quality_df))

    # --- 3. Save quality report CSV ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "03_data_quality_report.csv"
    quality_df.to_csv(report_path, index=False)
    logger.info("Quality report saved → %s", report_path)

    # --- 4. Missing data bar chart ---
    missing_data = (
        quality_df[quality_df["missing_pct"] > 0]
        .sort_values("missing_pct", ascending=False)
        .head(40)   # cap at 40 features for readability
    )

    if not missing_data.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = missing_data["source_file"] + "/" + missing_data["feature"]
        ax.barh(labels, missing_data["missing_pct"], color="#d94f3d")
        ax.set_xlabel("Missing (%)")
        ax.set_title(f"Features with Missing Values — {DATASET_NAME.upper()}")
        ax.invert_yaxis()
        plt.tight_layout()
        save_figure(fig, OUTPUT_DIR, "03_missing_data_barchart.svg")
        logger.info("Missing data bar chart saved.")
    else:
        logger.info("No missing data found — bar chart skipped.")

    # --- 5. Outlier count bar chart ---
    outlier_data = (
        quality_df[quality_df["n_outliers"].notna() & (quality_df["n_outliers"] > 0)]
        .sort_values("n_outliers", ascending=False)
        .head(40)
    )

    if not outlier_data.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = outlier_data["source_file"] + "/" + outlier_data["feature"]
        ax.barh(labels, outlier_data["n_outliers"], color="#3d7dd9")
        ax.set_xlabel("Outlier Count (IQR method)")
        ax.set_title(f"Outlier Counts per Feature — {DATASET_NAME.upper()}")
        ax.invert_yaxis()
        plt.tight_layout()
        save_figure(fig, OUTPUT_DIR, "03_outlier_counts.svg")
        logger.info("Outlier count bar chart saved.")
    else:
        logger.info("No outliers detected — bar chart skipped.")

    logger.info("Stage 03 complete.")


if __name__ == "__main__":
    main()
