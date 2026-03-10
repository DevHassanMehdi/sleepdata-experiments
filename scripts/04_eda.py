# =============================================================================
# scripts/04_eda.py
# -----------------------------------------------------------------------------
# Stage 04: Exploratory Data Analysis (EDA)
#
# What it does:
#   - Loads all CSVs from the dataset folder
#   - Per numeric column: histogram with KDE overlay
#   - Full correlation heatmap across all numeric columns
#   - Dimensionality reduction plots (PCA, t-SNE, UMAP), coloured by label
#     column if one is detected
#
# Outputs (in outputs/<DATASET_NAME>/04_eda/):
#   histograms/<file>_<column>_hist.svg   — one per numeric column
#   04_correlation_heatmap.svg
#   04_pca_plot.svg
#   04_tsne_plot.svg
#   04_umap_plot.svg
#
# Log file:
#   logs/04_eda_<DATASET_NAME>.txt
#
# Usage:
#   python scripts/04_eda.py
#   Change DATASET_NAME below to switch datasets.
# =============================================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

# Max rows used for dimensionality reduction (keeps runtime manageable)
DIM_REDUCTION_SAMPLE = 5000

# Keywords that identify a label/target column (from stage 02)
LABEL_KEYWORDS = [
    "label", "class", "target", "stage", "event", "diagnosis",
    "outcome", "category", "type", "status", "condition",
]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logger     = get_logger("04_eda", DATASET_NAME)
OUTPUT_DIR = config.OUTPUTS_DIR / DATASET_NAME / "04_eda"
HIST_DIR   = OUTPUT_DIR / "histograms"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_label_column(df):
    """Return the first column whose name contains a label keyword, or None."""
    for col in df.columns:
        if any(kw in col.lower() for kw in LABEL_KEYWORDS):
            return col
    return None


def build_combined_numeric(dataframes):
    """Merge numeric columns from all DataFrames; return (X, label_series)."""
    numeric_parts = []
    label_series  = None

    for filename, df in dataframes.items():
        # Detect label column
        if label_series is None:
            lc = find_label_column(df)
            if lc:
                label_series = df[lc].reset_index(drop=True)
                logger.info("Label column detected: '%s' in '%s'", lc, filename)

        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            numeric_parts.append(numeric_df.reset_index(drop=True))

    if not numeric_parts:
        return pd.DataFrame(), None

    combined = pd.concat(numeric_parts, axis=1)
    combined = combined.dropna(axis=1, how="all")   # drop all-NaN columns
    return combined, label_series


def prepare_matrix(combined_df, label_series, n_sample):
    """Return (X_scaled, labels) ready for dimensionality reduction."""
    df = combined_df.copy()

    # Align label series length
    if label_series is not None:
        min_len = min(len(df), len(label_series))
        df = df.iloc[:min_len]
        labels = label_series.iloc[:min_len]
    else:
        labels = None

    # Drop columns that are still NaN-heavy, fill remaining NaNs with median
    thresh = 0.5
    df = df.loc[:, df.isnull().mean() < thresh]
    df = df.fillna(df.median(numeric_only=True))

    if df.empty or df.shape[1] == 0:
        return None, None

    # Sample
    if len(df) > n_sample:
        idx    = np.random.RandomState(config.RANDOM_SEED).choice(len(df), n_sample, replace=False)
        df     = df.iloc[idx]
        labels = labels.iloc[idx] if labels is not None else None

    scaler = StandardScaler()
    X      = scaler.fit_transform(df.values)
    return X, labels


def scatter_plot(coords, labels, title, ax=None):
    """2-D scatter plot, optionally coloured by *labels*."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.get_figure()

    if labels is not None:
        unique_labels = sorted(labels.unique())
        palette       = sns.color_palette("tab10", len(unique_labels))
        label_to_color = dict(zip(unique_labels, palette))
        colors = labels.map(label_to_color)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.6)

        # Legend patches
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=str(l)) for l, c in label_to_color.items()]
        ax.legend(handles=handles, title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.5, color="#4472C4")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_thesis_style()
    np.random.seed(config.RANDOM_SEED)

    # --- 1. Load data ---
    data_path = get_dataset_path(DATASET_NAME)
    logger.info("Dataset path: %s", data_path)

    dataframes = load_csvs(data_path)

    if not dataframes:
        logger.warning("No data found. Run the appropriate downloader first.")
        return

    HIST_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. Histograms with KDE ---
    logger.info("Generating histograms...")
    for filename, df in dataframes.items():
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 2:
                continue
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(series, bins=50, density=True, alpha=0.5,
                        color="#4472C4", label="Histogram")
                series.plot.kde(ax=ax, color="#C00000", linewidth=2, label="KDE")
                ax.set_title(f"{filename} — {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Density")
                ax.legend()
                safe_col = col.replace("/", "_").replace(" ", "_")
                save_figure(fig, HIST_DIR, f"{filename}_{safe_col}_hist.svg")
            except Exception as exc:
                logger.warning("Histogram failed for %s/%s: %s", filename, col, exc)

    # --- 3. Correlation heatmap ---
    logger.info("Generating correlation heatmap...")
    combined, label_series = build_combined_numeric(dataframes)

    if not combined.empty and combined.shape[1] > 1:
        # Limit to first 50 numeric columns for readability
        plot_cols = combined.columns[:50]
        corr      = combined[plot_cols].corr()

        size = max(12, len(plot_cols) * 0.4)
        fig, ax = plt.subplots(figsize=(size, size * 0.8))
        sns.heatmap(
            corr, annot=False, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=ax,
            xticklabels=True, yticklabels=True,
        )
        ax.set_title(f"Correlation Matrix — {DATASET_NAME.upper()}")
        plt.tight_layout()
        save_figure(fig, OUTPUT_DIR, "04_correlation_heatmap.svg")
        logger.info("Correlation heatmap saved.")
    else:
        logger.warning("Not enough numeric data for correlation heatmap.")

    # --- 4. Dimensionality reduction ---
    X, labels = prepare_matrix(combined, label_series, DIM_REDUCTION_SAMPLE)

    if X is None or X.shape[0] < 10:
        logger.warning("Not enough data for dimensionality reduction — skipping.")
        logger.info("Stage 04 complete.")
        return

    # PCA
    logger.info("Running PCA...")
    pca_coords = PCA(n_components=2, random_state=config.RANDOM_SEED).fit_transform(X)
    fig = scatter_plot(pca_coords, labels, f"PCA — {DATASET_NAME.upper()}")
    save_figure(fig, OUTPUT_DIR, "04_pca_plot.svg")
    logger.info("PCA plot saved.")

    # t-SNE
    logger.info("Running t-SNE (this may take a moment)...")
    try:
        tsne_coords = TSNE(
            n_components=2, random_state=config.RANDOM_SEED, perplexity=30
        ).fit_transform(X)
        fig = scatter_plot(tsne_coords, labels, f"t-SNE — {DATASET_NAME.upper()}")
        save_figure(fig, OUTPUT_DIR, "04_tsne_plot.svg")
        logger.info("t-SNE plot saved.")
    except Exception as exc:
        logger.warning("t-SNE failed: %s", exc)

    # UMAP
    logger.info("Running UMAP...")
    try:
        import umap  # noqa: PLC0415 — lazy import; optional dependency
        umap_coords = umap.UMAP(
            n_components=2, random_state=config.RANDOM_SEED
        ).fit_transform(X)
        fig = scatter_plot(umap_coords, labels, f"UMAP — {DATASET_NAME.upper()}")
        save_figure(fig, OUTPUT_DIR, "04_umap_plot.svg")
        logger.info("UMAP plot saved.")
    except ImportError:
        logger.warning("umap-learn not installed. Run: pip install umap-learn")
    except Exception as exc:
        logger.warning("UMAP failed: %s", exc)

    logger.info("Stage 04 complete.")


if __name__ == "__main__":
    main()
