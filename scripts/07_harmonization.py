# =============================================================================
# scripts/07_harmonization.py
# -----------------------------------------------------------------------------
# Stage 07: Data Harmonization Across Datasets
#
# Purpose:
#   Apply harmonization methods to reduce systematic differences (batch effects)
#   between clinical PSG datasets and the wearable TIHM dataset, enabling fair
#   cross-device comparison in stage 08.
#
# Planned outputs (in outputs/harmonized/ or per-dataset):
#   07_harmonized_<dataset>.csv          — harmonized feature matrix per dataset
#   07_harmonization_comparison.svg      — before/after PCA plots
#   07_batch_effect_report.csv           — per-feature variance explained by batch
#
# Log file:
#   logs/07_harmonization_all.txt
#
# Usage (when implemented):
#   python3 scripts/07_harmonization.py
#   (Runs across all datasets — no per-dataset flag needed)
# =============================================================================

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# TODO: Step 1 — Load aligned feature sets
# -----------------------------------------------------------------------------
# - Load the shared feature subset identified in stage 05
#   (outputs/05_feature_alignment_matrix.csv)
# - For each dataset, load only the shared features
# - Concatenate into a single matrix with a 'dataset' batch column
# - Log feature counts and sample counts per dataset
# =============================================================================

# =============================================================================
# TODO: Step 2 — Quantify batch effects (pre-harmonization)
# -----------------------------------------------------------------------------
# - Run PCA on the combined matrix, colour points by dataset
# - Compute per-feature ANOVA across datasets to measure batch variance
# - Log and save 07_batch_effect_report.csv
# =============================================================================

# =============================================================================
# TODO: Step 3 — Apply harmonization methods
# -----------------------------------------------------------------------------
# Candidate approaches (evaluate at least two):
#   a) ComBat harmonization (neuroCombat or similar)
#      — Adjusts for known batch (dataset) effects while preserving biology
#   b) Z-score normalisation per dataset (simple baseline)
#   c) Quantile normalisation
#   d) Domain-adversarial neural network feature alignment (advanced)
# For each method: fit on PSG datasets, apply to TIHM
# =============================================================================

# =============================================================================
# TODO: Step 4 — Validate harmonization
# -----------------------------------------------------------------------------
# - Re-run PCA post-harmonization, colour by dataset (batch effect reduced?)
# - Re-run PCA, colour by label (biological signal preserved?)
# - Compute post-harmonization ANOVA batch variance
# - Save before/after comparison plots as SVG
# =============================================================================

# =============================================================================
# TODO: Step 5 — Export harmonized datasets
# -----------------------------------------------------------------------------
# - Save 07_harmonized_<dataset>.csv for each dataset
# - These files are the inputs to stage 08 model training
# - Log sample counts and feature counts for each exported file
# =============================================================================


def main():
    raise NotImplementedError(
        "Stage 07 is not yet implemented. "
        "See TODO sections in this file for the planned steps."
    )


if __name__ == "__main__":
    main()
