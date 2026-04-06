# =============================================================================
# scripts/06_model_training.py
# -----------------------------------------------------------------------------
# Stage 06: Machine Learning Model Training (Per Dataset)
#
# Purpose:
#   Train and evaluate a suite of ML classifiers on each individual dataset
#   using the features identified in stages 01–05.  Results feed into stage 08
#   for cross-dataset comparison.
#
# Planned outputs (in outputs/<DATASET_NAME>/):
#   06_model_results.csv         — per-model metrics (accuracy, F1, AUC, etc.)
#   06_roc_curves.svg            — ROC curves for each model
#   06_confusion_matrices.svg    — confusion matrix grid
#   06_feature_importances.svg   — feature importance (for tree-based models)
#
# Log file:
#   logs/06_model_training_<DATASET_NAME>.txt
#
# Usage (when implemented):
#   python3 scripts/06_model_training.py                  # all datasets
#   python3 scripts/06_model_training.py --dataset tihm   # one dataset
# =============================================================================

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config

# =============================================================================
# TODO: Step 1 — Data loading and label extraction
# -----------------------------------------------------------------------------
# - Use utils/data_loader.py to load CSVs from data/<DATASET_NAME>/
# - Identify the target column using label keywords (same approach as stage 04)
# - Combine and align feature columns from all CSV files
# - Log dataset shape and class distribution
# =============================================================================

# =============================================================================
# TODO: Step 2 — Preprocessing pipeline
# -----------------------------------------------------------------------------
# - Drop columns with > 50% missing values
# - Impute remaining missing values (median for numeric, mode for categorical)
# - Encode categorical features (one-hot or ordinal as appropriate)
# - Scale numeric features (StandardScaler or MinMaxScaler)
# - Apply feature selection from stage 02 inventory if applicable
# - Split data: stratified train/test (80/20), reproducible with RANDOM_SEED
# =============================================================================

# =============================================================================
# TODO: Step 3 — Model definition
# -----------------------------------------------------------------------------
# Candidate models to evaluate:
#   - Logistic Regression (baseline)
#   - Random Forest
#   - Gradient Boosting (XGBoost / LightGBM)
#   - Support Vector Machine (RBF kernel)
#   - k-Nearest Neighbours
# Wrap each in sklearn Pipeline with preprocessing steps.
# =============================================================================

# =============================================================================
# TODO: Step 4 — Cross-validation and evaluation
# -----------------------------------------------------------------------------
# - Stratified k-fold CV (k=5), reproducible with RANDOM_SEED
# - Metrics to compute per fold: accuracy, macro F1, weighted F1, ROC-AUC
# - Use utils/metrics.py: classification_report_df(), roc_auc_score_multiclass()
# - Aggregate CV results (mean ± std)
# - Final refit on full training set, evaluate on held-out test set
# =============================================================================

# =============================================================================
# TODO: Step 5 — Save results and visualisations
# -----------------------------------------------------------------------------
# - Save 06_model_results.csv with all model × metric combinations
# - Plot ROC curves (one per model, multi-class OvR if needed)
# - Plot confusion matrices as a subplot grid
# - Plot feature importances for tree-based models
# - Use utils/plotting.py for all figures
# =============================================================================


def main(dataset_name):
    raise NotImplementedError(
        f"Stage 06 is not yet implemented for dataset '{dataset_name}'. "
        "See TODO sections in this file for the planned steps."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 06: Model Training")
    parser.add_argument(
        "--dataset", metavar="NAME",
        help="Dataset to process (default: all). One of: " + ", ".join(config.DATASET_PATHS),
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(config.DATASET_PATHS.keys())

    for ds in datasets:
        print(f"\n{'='*50}\n  {ds.upper()}\n{'='*50}")
        main(ds)
