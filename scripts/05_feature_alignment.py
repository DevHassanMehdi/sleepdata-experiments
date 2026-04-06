# =============================================================================
# scripts/05_feature_alignment.py
# -----------------------------------------------------------------------------
# Stage 05: Cross-Dataset Feature Alignment
#
# What it does:
#   - Loads the 02_feature_inventory.csv output from EVERY dataset
#   - Builds a feature alignment matrix:
#       rows    = unique feature names across all datasets
#       columns = dataset names
#       values  = True/False (feature present in that dataset)
#   - Identifies shared features, PSG-only features, TIHM-only features
#   - Saves the alignment matrix as CSV and a heatmap as SVG
#
# Prerequisite:
#   Run scripts/02_feature_inventory.py for all datasets first.
#
# Outputs (in outputs/   — root outputs, NOT per-dataset):
#   05_feature_alignment_matrix.csv
#   05_feature_alignment_heatmap.svg
#
# Log file:
#   logs/05_feature_alignment_all.txt
#
# Usage:
#   python3 scripts/05_feature_alignment.py
#   (Runs across all datasets — no per-dataset flag needed)
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
from utils.logger import get_logger
from utils.plotting import save_figure, set_thesis_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALL_DATASETS = ["tihm", "shhs", "mros", "mesa", "ssc"]
PSG_DATASETS = ["shhs", "mros", "mesa", "ssc"]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logger     = get_logger("05_feature_alignment", "all")
OUTPUT_DIR = config.OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_thesis_style()

    # --- 1. Load feature inventories ---
    dataset_features = {}  # dataset_name -> set of feature names

    for dataset in ALL_DATASETS:
        inventory_path = config.OUTPUTS_DIR / dataset / "02_feature_inventory.csv"

        if not inventory_path.exists():
            logger.warning(
                "Inventory not found for '%s' at %s — skipping.", dataset, inventory_path
            )
            continue

        inv_df = pd.read_csv(inventory_path)
        features = set(inv_df["feature_name"].str.lower().str.strip().unique())
        dataset_features[dataset] = features
        logger.info("Loaded %d features for dataset: %s", len(features), dataset)

    if len(dataset_features) < 2:
        logger.error(
            "Need at least 2 datasets with inventory files. "
            "Run 02_feature_inventory.py for each dataset first."
        )
        return

    available_datasets = list(dataset_features.keys())

    # --- 2. Build alignment matrix ---
    all_features = sorted(set().union(*dataset_features.values()))
    logger.info("Total unique features across all datasets: %d", len(all_features))

    alignment = pd.DataFrame(
        index=all_features,
        columns=available_datasets,
        dtype=bool,
    )
    alignment[:] = False

    for dataset, features in dataset_features.items():
        for feat in features:
            if feat in alignment.index:
                alignment.loc[feat, dataset] = True

    # --- 3. Analyse shared vs dataset-specific features ---
    available_psg     = [d for d in PSG_DATASETS if d in dataset_features]
    available_non_psg = [d for d in available_datasets if d not in PSG_DATASETS]

    shared_all  = alignment[alignment.all(axis=1)].index.tolist()
    psg_only    = []
    tihm_only   = []

    if available_psg and available_non_psg:
        psg_mask    = alignment[available_psg].all(axis=1)
        tihm_mask   = alignment[available_non_psg].all(axis=1)
        psg_only    = alignment[psg_mask & ~tihm_mask].index.tolist()
        tihm_only   = alignment[tihm_mask & ~psg_mask].index.tolist()

    logger.info("Features shared across ALL datasets (%d): %s", len(shared_all), shared_all)
    logger.info("PSG-only features (%d): %s", len(psg_only), psg_only)
    logger.info("TIHM-only features (%d): %s", len(tihm_only), tihm_only)

    # --- 4. Save alignment matrix CSV ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = OUTPUT_DIR / "05_feature_alignment_matrix.csv"
    alignment.to_csv(matrix_path)
    logger.info("Alignment matrix saved → %s", matrix_path)

    # --- 5. Heatmap ---
    logger.info("Generating alignment heatmap...")

    # Convert bool to int for heatmap colouring
    heatmap_data = alignment.astype(int)

    # Limit rows if very large for readability
    MAX_ROWS = 80
    if len(heatmap_data) > MAX_ROWS:
        logger.info(
            "Truncating heatmap to top %d features (by presence count).", MAX_ROWS
        )
        row_counts   = heatmap_data.sum(axis=1).sort_values(ascending=False)
        heatmap_data = heatmap_data.loc[row_counts.head(MAX_ROWS).index]

    fig_height = max(8, len(heatmap_data) * 0.2)
    fig, ax    = plt.subplots(figsize=(max(8, len(available_datasets) * 1.5), fig_height))

    sns.heatmap(
        heatmap_data,
        cmap=["#f0f0f0", "#2E75B6"],
        linewidths=0.4,
        linecolor="white",
        cbar=False,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        annot=True,
        fmt="d",
    )
    ax.set_title("Feature Alignment Across Datasets", fontsize=14)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "05_feature_alignment_heatmap.svg")
    logger.info("Alignment heatmap saved.")

    logger.info("Stage 05 complete.")


if __name__ == "__main__":
    main()
