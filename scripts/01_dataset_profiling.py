# =============================================================================
# scripts/01_dataset_profiling.py
# -----------------------------------------------------------------------------
# Stage 01: Dataset Profiling
#
# What it does:
#   - Loads all CSV files from the target dataset folder
#   - Logs per-file: shape, column names, dtypes, missing value counts,
#     and descriptive statistics
#   - Saves a missing values heatmap (SVG)
#   - Saves a summary of all column names across all files (CSV)
#
# Outputs (in outputs/<DATASET_NAME>/):
#   01_missing_values_heatmap.svg
#   01_column_summary.csv
#
# Log file:
#   logs/01_dataset_profiling_<DATASET_NAME>.txt
#
# Usage:
#   python scripts/01_dataset_profiling.py
#   Change DATASET_NAME below to switch datasets.
# =============================================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
logger     = get_logger("01_dataset_profiling", DATASET_NAME)
OUTPUT_DIR = config.OUTPUTS_DIR / DATASET_NAME


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_thesis_style()

    # --- 1. Resolve dataset path and load CSVs ---
    data_path = get_dataset_path(DATASET_NAME)
    logger.info("Dataset path: %s", data_path)

    dataframes = load_csvs(data_path)

    if not dataframes:
        logger.warning("No data found. Run the appropriate downloader first.")
        return

    # --- 2. Per-file profiling ---
    column_records = []

    for filename, df in dataframes.items():
        logger.info("=" * 60)
        logger.info("File: %s", filename)
        logger.info("Shape: %d rows × %d columns", *df.shape)
        logger.info("Columns: %s", df.columns.tolist())

        # dtypes
        logger.info("Data types:\n%s", df.dtypes.to_string())

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        logger.info(
            "Missing values:\n%s",
            pd.concat([missing, missing_pct], axis=1, keys=["count", "pct"]).to_string(),
        )

        # Descriptive statistics (numeric columns only)
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            logger.info("Descriptive statistics:\n%s", numeric_df.describe().to_string())

        # Collect column metadata for summary
        for col in df.columns:
            column_records.append({
                "source_file": filename,
                "column_name": col,
                "dtype":       str(df[col].dtype),
                "missing_count": int(df[col].isnull().sum()),
                "missing_pct":   round(df[col].isnull().sum() / len(df) * 100, 2),
            })

    # --- 3. Missing values heatmap ---
    logger.info("Generating missing values heatmap...")

    # Build a combined missing indicator matrix (sample up to 1000 rows per file)
    missing_frames = []
    for filename, df in dataframes.items():
        sample = df.head(1000) if len(df) > 1000 else df
        m = sample.isnull().astype(int)
        m.columns = [f"{filename}/{c}" for c in m.columns]
        missing_frames.append(m)

    if missing_frames:
        combined_missing = pd.concat(missing_frames, axis=1).fillna(0)

        # Only include columns that have at least one missing value
        has_missing = combined_missing.columns[combined_missing.sum() > 0]
        if len(has_missing) > 0:
            plot_data = combined_missing[has_missing]

            fig, ax = plt.subplots(figsize=(max(14, len(has_missing) * 0.4), 6))
            sns.heatmap(
                plot_data.T,
                cbar=False,
                cmap="Reds",
                ax=ax,
                yticklabels=True,
                xticklabels=False,
            )
            ax.set_title(f"Missing Values — {DATASET_NAME.upper()}", fontsize=14)
            ax.set_xlabel("Samples (rows)")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            save_figure(fig, OUTPUT_DIR, "01_missing_values_heatmap.svg")
        else:
            logger.info("No missing values found — heatmap skipped.")

    # --- 4. Column summary CSV ---
    summary_df = pd.DataFrame(column_records)
    out_path   = OUTPUT_DIR / "01_column_summary.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    logger.info("Column summary saved → %s", out_path)
    logger.info("Total columns catalogued: %d", len(summary_df))

    logger.info("Stage 01 complete.")


if __name__ == "__main__":
    main()
