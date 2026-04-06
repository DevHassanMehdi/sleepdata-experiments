# =============================================================================
# scripts/08_comparison.py
# -----------------------------------------------------------------------------
# Stage 08: Cross-Dataset Performance Comparison
#
# Purpose:
#   Aggregate results from stage 06 (per-dataset model training) and stage 07
#   (harmonization), then produce a comprehensive comparison showing whether
#   data harmonization improves cross-device ML generalisation.
#
# Planned outputs (in outputs/comparison/):
#   08_performance_comparison.csv         — full model × dataset × condition table
#   08_performance_heatmap.svg            — metric heatmap across all conditions
#   08_boxplots_by_dataset.svg            — per-dataset metric distributions
#   08_boxplots_by_model.svg              — per-model metric distributions
#   08_harmonization_delta.csv            — per-metric improvement from harmonization
#   08_statistical_tests.csv             — pairwise significance results
#   08_final_summary_figure.svg           — thesis-quality multi-panel summary
#
# Log file:
#   logs/08_comparison_all.txt
#
# Usage (when implemented):
#   python3 scripts/08_comparison.py
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
# TODO: Step 1 — Load stage 06 results (pre-harmonization)
# -----------------------------------------------------------------------------
# - Load outputs/<dataset>/06_model_results.csv for all datasets
# - Concatenate into a single DataFrame with 'dataset' and 'condition' columns
# - Condition label: "original" for pre-harmonization results
# =============================================================================

# =============================================================================
# TODO: Step 2 — Load stage 06 results on harmonized data (post-harmonization)
# -----------------------------------------------------------------------------
# - Re-run (or load cached results of) 06_model_training.py on the
#   harmonized CSVs produced by stage 07
# - Concatenate with condition label: "harmonized"
# =============================================================================

# =============================================================================
# TODO: Step 3 — Build comparison tables
# -----------------------------------------------------------------------------
# - Merge original vs harmonized results
# - Compute delta metrics (harmonized - original) per model × dataset
# - Save 08_performance_comparison.csv and 08_harmonization_delta.csv
# =============================================================================

# =============================================================================
# TODO: Step 4 — Statistical significance testing
# -----------------------------------------------------------------------------
# - Paired Wilcoxon signed-rank test: original vs harmonized per metric
# - Friedman test across datasets per model
# - Apply Bonferroni correction for multiple comparisons
# - Save 08_statistical_tests.csv with p-values and effect sizes
# =============================================================================

# =============================================================================
# TODO: Step 5 — Visualisations
# -----------------------------------------------------------------------------
# - Performance heatmap: rows = model×dataset, columns = metrics
# - Box plots grouped by dataset (one panel per metric)
# - Box plots grouped by model (one panel per metric)
# - Multi-panel summary figure for thesis (publication quality)
# - Use utils/plotting.py for all figures; save as SVG
# =============================================================================

# =============================================================================
# TODO: Step 6 — Narrative summary
# -----------------------------------------------------------------------------
# - Log key findings: best model overall, biggest harmonization gains,
#   datasets with highest/lowest cross-device generalisation
# - Write a brief text summary to outputs/comparison/08_key_findings.txt
# =============================================================================


def main():
    raise NotImplementedError(
        "Stage 08 is not yet implemented. "
        "See TODO sections in this file for the planned steps."
    )


if __name__ == "__main__":
    main()
