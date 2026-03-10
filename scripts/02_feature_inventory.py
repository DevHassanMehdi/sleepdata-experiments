# =============================================================================
# scripts/02_feature_inventory.py
# -----------------------------------------------------------------------------
# Stage 02: Feature Inventory
#
# What it does:
#   - Loads all CSVs from the dataset folder
#   - Extracts every column name from every file
#   - Categorises each column using keyword matching into:
#       demographic, physiological, temporal, label/target, unknown
#   - Saves a full feature inventory CSV with category, dtype, missing%
#
# Outputs (in outputs/<DATASET_NAME>/):
#   02_feature_inventory.csv
#       Columns: feature_name, source_file, inferred_category, dtype, missing_pct
#
# Log file:
#   logs/02_feature_inventory_<DATASET_NAME>.txt
#
# Usage:
#   python scripts/02_feature_inventory.py
#   Change DATASET_NAME below to switch datasets.
# =============================================================================

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from utils.data_loader import get_dataset_path, load_csvs
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Configuration — change this to switch datasets
# ---------------------------------------------------------------------------
DATASET_NAME = "tihm"

# ---------------------------------------------------------------------------
# Keyword mapping for category inference
# Keywords are matched against lowercase column names (substring match)
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "demographic": [
        "age", "sex", "gender", "bmi", "height", "weight", "race",
        "ethnicity", "education", "marital", "employment", "income",
    ],
    "physiological": [
        "hr", "spo2", "eeg", "eog", "emg", "ecg", "resp", "oximetry",
        "pulse", "oxygen", "apnea", "ahi", "sdb", "arousal", "flow",
        "pressure", "temp", "saturation", "frequency", "amplitude",
    ],
    "temporal": [
        "time", "date", "timestamp", "epoch", "duration", "start", "end",
        "onset", "offset", "interval", "period", "night", "morning",
    ],
    "label": [
        "label", "class", "target", "stage", "event", "diagnosis",
        "outcome", "category", "type", "status", "condition", "score",
    ],
}


def infer_category(column_name):
    """Return the best-matching category for a column name via keyword search.

    Parameters
    ----------
    column_name : str

    Returns
    -------
    str
        One of: ``demographic``, ``physiological``, ``temporal``,
        ``label``, ``unknown``.
    """
    col_lower = column_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in col_lower for kw in keywords):
            return category
    return "unknown"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logger     = get_logger("02_feature_inventory", DATASET_NAME)
OUTPUT_DIR = config.OUTPUTS_DIR / DATASET_NAME


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- 1. Load data ---
    data_path = get_dataset_path(DATASET_NAME)
    logger.info("Dataset path: %s", data_path)

    dataframes = load_csvs(data_path)

    if not dataframes:
        logger.warning("No data found. Run the appropriate downloader first.")
        return

    # --- 2. Build feature inventory ---
    records = []

    for filename, df in dataframes.items():
        for col in df.columns:
            missing_pct = round(df[col].isnull().sum() / len(df) * 100, 2)
            records.append({
                "feature_name":      col,
                "source_file":       filename,
                "inferred_category": infer_category(col),
                "dtype":             str(df[col].dtype),
                "missing_pct":       missing_pct,
            })

    inventory_df = pd.DataFrame(records)
    logger.info("Total features catalogued: %d", len(inventory_df))

    # --- 3. Log category summary ---
    category_counts = inventory_df["inferred_category"].value_counts()
    logger.info("Category breakdown:\n%s", category_counts.to_string())

    # --- 4. Save inventory CSV ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "02_feature_inventory.csv"
    inventory_df.to_csv(out_path, index=False)
    logger.info("Feature inventory saved → %s", out_path)

    logger.info("Stage 02 complete.")


if __name__ == "__main__":
    main()
