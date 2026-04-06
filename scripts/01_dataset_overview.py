"""
scripts/01_dataset_overview.py
===============================
Stage 01: Dataset Overview

Loads the four key dataset files (SHHS, MrOS, MESA harmonized PSG datasets and
TIHM Sleep wearable data), profiles each one, builds a cross-dataset feature
alignment map, and generates a set of publication-ready visualisations.

Also loads supplementary columns (heart rate, snoring) from full PSG dataset
files to extend the feature alignment map beyond the harmonized variable set.

Outputs (all saved to outputs/overview/):
  CSV files:
    shhs_stats.csv                    — descriptive statistics for SHHS
    mros_stats.csv                    — descriptive statistics for MrOS
    mesa_stats.csv                    — descriptive statistics for MESA
    tihm_sleep_stats.csv              — descriptive statistics for TIHM Sleep
    feature_alignment_matrix_v3.csv   — direct/derivable/none presence matrix

  SVG plots:
    02_feature_overlap_heatmap.svg    — category-grouped heatmap (direct/derivable/none)
    03_shared_feature_distributions.svg — KDE overlay for shared PSG features
    04_correlation_heatmap_shhs.svg   — correlation matrix for SHHS harmonized data
    04_correlation_heatmap_mros.svg   — correlation matrix for MrOS harmonized data
    04_correlation_heatmap_mesa.svg   — correlation matrix for MESA harmonized data
    05_tihm_distributions.svg         — distribution plots for TIHM Sleep features
    06_tihm_sleep_over_time.svg       — longitudinal HR + RR dual-axis plot for one TIHM patient
    07_aligned_feature_summary.svg    — table of features available in TIHM + at least one PSG

Log file:
  logs/01_dataset_overview_all.txt

Usage:
  python scripts/01_dataset_overview.py
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import get_logger
from utils.plotting import save_figure

# =============================================================================
# UNIFORM THESIS STYLE
# Applied once before any plot is generated.
# =============================================================================

def apply_thesis_style():
    mpl.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.grid":            True,
        "grid.color":           "#E5E5E5",
        "grid.linestyle":       "-",
        "grid.linewidth":       0.7,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     True,
        "axes.spines.bottom":   True,
        "axes.edgecolor":       "#CCCCCC",
        "axes.labelcolor":      "#333333",
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.titlepad":        12,
        "axes.labelsize":       11,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "xtick.color":          "#555555",
        "ytick.color":          "#555555",
        "legend.fontsize":      10,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
        "font.family":          "sans-serif",
        "text.color":           "#333333",
        "figure.dpi":           150,
    })


# =============================================================================
# CONFIGURABLE PATHS — edit here to point at different file versions
# =============================================================================

# Harmonized PSG files (primary source for per-night summary features)
SHHS_HARMONIZED = ROOT_DIR / "data/shhs/datasets/shhs-harmonized-dataset-0.21.0.csv"
MROS_HARMONIZED = ROOT_DIR / "data/mros/datasets/mros-visit1-harmonized-0.6.0.csv"
MESA_HARMONIZED = ROOT_DIR / "data/mesa/datasets/mesa-sleep-harmonized-dataset-0.8.0.csv"

# Full PSG files (used to check availability of HR / snoring columns not in harmonized)
SHHS_FULL = ROOT_DIR / "data/shhs/datasets/shhs1-dataset-0.21.0.csv"
MROS_FULL = ROOT_DIR / "data/mros/datasets/mros-visit1-dataset-0.6.0.csv"
MESA_FULL = ROOT_DIR / "data/mesa/datasets/mesa-sleep-dataset-0.8.0.csv"

# TIHM wearable IoT data
TIHM_SLEEP = ROOT_DIR / "data/tihm/Sleep.csv"

OUTPUT_DIR  = ROOT_DIR / "outputs/overview"
LOG_FILE    = ROOT_DIR / "logs/01_dataset_overview.txt"
PLOT_FORMAT = "svg"

# ---------------------------------------------------------------------------
# Columns to load from each full PSG file (only what we need)
# ---------------------------------------------------------------------------
FULL_PSG_COLS = {
    "shhs": {"path": SHHS_FULL, "cols": ["avhrbp",    "mnhrbp",   "mxhrbp",   "hvsnrd02"]},
    "mros": {"path": MROS_FULL, "cols": ["pobasehrt", "pobpmmin", "pobpmmax",  "slsnore"]},
    "mesa": {"path": MESA_FULL, "cols": ["bpmavg5",   "bpmmin5",  "bpmmax5",   "snored5"]},
}

# Subject ID column per dataset
SUBJECT_ID = {
    "shhs": "nsrrid",
    "mros": "nsrrid",
    "mesa": "nsrrid",
    "tihm": "patient_id",
}

# ---------------------------------------------------------------------------
# Consistent color palette used across ALL plots
# ---------------------------------------------------------------------------
DATASET_COLORS = {
    "shhs": "#2196F3",   # blue
    "mros": "#4CAF50",   # green
    "mesa": "#FF9800",   # orange
    "tihm": "#9C27B0",   # purple
}

# =============================================================================
# FEATURE ALIGNMENT MAP (column name lookup — used for data loading / profiling)
# =============================================================================
FEATURE_MAP = {
    # --- Demographics ---
    "age":               {"shhs": "nsrr_age",           "mros": "nsrr_age",           "mesa": "nsrr_age",           "tihm": None},
    "sex":               {"shhs": "nsrr_sex",           "mros": "nsrr_sex",           "mesa": "nsrr_sex",           "tihm": None},
    "bmi":               {"shhs": "nsrr_bmi",           "mros": "nsrr_bmi",           "mesa": "nsrr_bmi",           "tihm": None},
    "bp_systolic":       {"shhs": "nsrr_bp_systolic",   "mros": "nsrr_bp_systolic",   "mesa": None,                 "tihm": None},
    "bp_diastolic":      {"shhs": "nsrr_bp_diastolic",  "mros": "nsrr_bp_diastolic",  "mesa": None,                 "tihm": None},
    # --- Sleep-disordered breathing ---
    "ahi":               {"shhs": "nsrr_ahi_hp3u",      "mros": "nsrr_ahi_hp3u",      "mesa": "nsrr_ahi_hp3u",      "tihm": None},
    # --- Sleep architecture ---
    "total_sleep_time":  {"shhs": "nsrr_ttldursp_f1",   "mros": "nsrr_ttldursp_f1",   "mesa": "nsrr_tst_f1",        "tihm": None},
    "sleep_efficiency":  {"shhs": "nsrr_ttleffsp_f1",   "mros": None,                 "mesa": "nsrr_ttleffsp_f1",   "tihm": None},
    "sleep_latency":     {"shhs": "nsrr_ttllatsp_f1",   "mros": None,                 "mesa": "nsrr_ttllatsp_f1",   "tihm": None},
    "arousal_index":     {"shhs": "nsrr_phrnumar_f1",   "mros": "nsrr_phrnumar_f1",   "mesa": "nsrr_phrnumar_f1",   "tihm": None},
    "pct_stage1":        {"shhs": "nsrr_pctdursp_s1",   "mros": None,                 "mesa": "nsrr_pctdursp_s1",   "tihm": None},
    "pct_stage2":        {"shhs": "nsrr_pctdursp_s2",   "mros": None,                 "mesa": "nsrr_pctdursp_s2",   "tihm": None},
    "pct_stage3":        {"shhs": "nsrr_pctdursp_s3",   "mros": None,                 "mesa": "nsrr_pctdursp_s3",   "tihm": None},
    "pct_rem":           {"shhs": "nsrr_pctdursp_sr",   "mros": None,                 "mesa": "nsrr_pctdursp_sr",   "tihm": None},
    "waso":              {"shhs": None,                  "mros": None,                 "mesa": "nsrr_waso_f1",       "tihm": None},
    # --- Oximetry ---
    "avg_spo2":          {"shhs": None,                  "mros": None,                 "mesa": "nsrr_avglvlsa",      "tihm": None},
    "min_spo2":          {"shhs": None,                  "mros": None,                 "mesa": "nsrr_minlvlsa",      "tihm": None},
    # --- Heart rate [FULL PSG files + TIHM] ---
    "heart_rate_avg":    {"shhs": "avhrbp",              "mros": "pobasehrt",          "mesa": "bpmavg5",            "tihm": "heart_rate"},
    "heart_rate_min":    {"shhs": "mnhrbp",              "mros": "pobpmmin",           "mesa": "bpmmin5",            "tihm": None},
    "heart_rate_max":    {"shhs": "mxhrbp",              "mros": "pobpmmax",           "mesa": "bpmmax5",            "tihm": None},
    # --- Snoring [FULL PSG files + TIHM] ---
    "snoring_flag":      {"shhs": "hvsnrd02",            "mros": "slsnore",            "mesa": "snored5",            "tihm": "snoring"},
    # --- TIHM-only wearable features ---
    "respiratory_rate":  {"shhs": None,                  "mros": None,                 "mesa": None,                 "tihm": "respiratory_rate"},
    "sleep_wake_state":  {"shhs": None,                  "mros": None,                 "mesa": None,                 "tihm": "state"},
}

PSG_DATASETS  = ["shhs", "mros", "mesa"]
DIST_FEATURES = ["total_sleep_time", "sleep_efficiency", "ahi", "arousal_index", "pct_rem", "age"]

# =============================================================================
# FEATURE MAP V3 — "direct" / "derivable" / None
# Used for visualisations and the v3 alignment matrix CSV.
# =============================================================================
FEATURE_MAP_V3 = {
    # Demographic
    "age":                  {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": "direct"},
    "sex":                  {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": "direct"},
    "bmi":                  {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": None},
    # Blood pressure
    "bp_systolic":          {"shhs": "direct",    "mros": "direct",    "mesa": None,        "tihm": "derivable"},
    "bp_diastolic":         {"shhs": "direct",    "mros": "direct",    "mesa": None,        "tihm": "derivable"},
    # Sleep breathing
    "ahi":                  {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": None},
    "arousal_index":        {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": "derivable"},
    # Sleep duration and quality
    "total_sleep_time":     {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": "derivable"},
    "sleep_efficiency":     {"shhs": "direct",    "mros": None,        "mesa": "direct",    "tihm": "derivable"},
    "sleep_latency":        {"shhs": "direct",    "mros": None,        "mesa": "direct",    "tihm": "derivable"},
    "waso":                 {"shhs": None,        "mros": None,        "mesa": "direct",    "tihm": "derivable"},
    # Sleep stages
    "pct_stage1":           {"shhs": "direct",    "mros": None,        "mesa": "direct",    "tihm": None},
    "pct_stage2":           {"shhs": "direct",    "mros": None,        "mesa": "direct",    "tihm": None},
    "pct_stage3":           {"shhs": "direct",    "mros": None,        "mesa": "direct",    "tihm": None},
    "pct_rem":              {"shhs": "direct",    "mros": None,        "mesa": "direct",    "tihm": None},
    # Oxygen
    "avg_spo2":             {"shhs": None,        "mros": None,        "mesa": "direct",    "tihm": None},
    "min_spo2":             {"shhs": None,        "mros": None,        "mesa": "direct",    "tihm": None},
    # Heart rate
    "heart_rate_avg":       {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": "direct"},
    "heart_rate_min":       {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": "derivable"},
    "heart_rate_max":       {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": "derivable"},
    "heart_rate_std":       {"shhs": None,        "mros": None,        "mesa": None,        "tihm": "derivable"},
    # Respiratory
    "respiratory_rate_avg": {"shhs": None,        "mros": None,        "mesa": None,        "tihm": "direct"},
    "respiratory_rate_std": {"shhs": None,        "mros": None,        "mesa": None,        "tihm": "derivable"},
    # Snoring
    "snoring_pct":          {"shhs": "direct",    "mros": "direct",    "mesa": "direct",    "tihm": "derivable"},
    # Sleep-wake
    "sleep_wake_state":     {"shhs": None,        "mros": None,        "mesa": None,        "tihm": "direct"},
}

# Ordered feature groupings for the heatmap plot
FEATURE_CATEGORIES = {
    "Demographic":              ["age", "sex", "bmi"],
    "Blood Pressure":           ["bp_systolic", "bp_diastolic"],
    "Sleep Breathing":          ["ahi", "arousal_index"],
    "Sleep Duration & Quality": ["total_sleep_time", "sleep_efficiency", "sleep_latency", "waso"],
    "Sleep Stages":             ["pct_stage1", "pct_stage2", "pct_stage3", "pct_rem"],
    "Oxygen":                   ["avg_spo2", "min_spo2"],
    "Heart Rate":               ["heart_rate_avg", "heart_rate_min", "heart_rate_max", "heart_rate_std"],
    "Respiratory":              ["respiratory_rate_avg", "respiratory_rate_std"],
    "Snoring":                  ["snoring_pct"],
    "Sleep-Wake":               ["sleep_wake_state"],
}

# Category lookup (feature → category name)
FEATURE_TO_CATEGORY = {
    feat: cat
    for cat, feats in FEATURE_CATEGORIES.items()
    for feat in feats
}

# How derivable TIHM features are computed
TIHM_DERIVATION_NOTES = {
    "age":            "From Demographics.csv",
    "sex":            "From Demographics.csv",
    "bp_systolic":    "From Physiology.csv if available",
    "bp_diastolic":   "From Physiology.csv if available",
    "arousal_index":  "Proxy via sleep-wake transitions",
    "total_sleep_time": "Sum of SLEEP epochs",
    "sleep_efficiency": "SLEEP mins / total mins × 100",
    "sleep_latency":    "Mins to first SLEEP epoch",
    "waso":             "Wake mins after first SLEEP",
    "heart_rate_min":   "Min of per-minute HR",
    "heart_rate_max":   "Max of per-minute HR",
    "heart_rate_std":   "Std dev of per-minute HR",
    "respiratory_rate_std": "Std dev of per-minute RR",
    "snoring_pct":      "% of mins with snoring=True",
}


# =============================================================================
# STEP 1 — Load datasets  [UNCHANGED]
# =============================================================================

def load_datasets(logger):
    """Load harmonized PSG files and TIHM Sleep.csv. Returns {name: DataFrame}."""
    sources = {
        "shhs": (SHHS_HARMONIZED, {}),
        "mros": (MROS_HARMONIZED, {}),
        "mesa": (MESA_HARMONIZED, {}),
        "tihm": (TIHM_SLEEP,      {"parse_dates": ["date"]}),
    }
    datasets = {}
    for name, (path, kwargs) in sources.items():
        logger.info("Loading %s from: %s", name.upper(), path)
        try:
            df = pd.read_csv(path, low_memory=False, **kwargs)
            logger.info("  Shape   : %d rows × %d columns", *df.shape)
            logger.info("  Columns : %s", df.columns.tolist())
            datasets[name] = df
        except FileNotFoundError:
            logger.error("  File not found — skipping %s: %s", name.upper(), path)
        except Exception as exc:
            logger.error("  Failed to load %s: %s", name.upper(), exc)

    logger.info("Loaded %d / 4 datasets.", len(datasets))
    return datasets


def load_full_psg_columns(logger):
    """
    Load only the supplementary HR and snoring columns from full PSG dataset files.
    These columns are not present in the harmonized files.
    Returns {dataset_name: DataFrame} containing only the columns that were found.
    """
    full_psg = {}
    for ds, spec in FULL_PSG_COLS.items():
        path    = spec["path"]
        wanted  = spec["cols"]
        logger.info("Loading full PSG columns for %s from: %s", ds.upper(), path)
        try:
            available = pd.read_csv(path, nrows=0).columns.tolist()
            found     = [c for c in wanted if c in available]
            missing   = [c for c in wanted if c not in available]
            if missing:
                logger.warning("  Columns NOT found in %s full dataset: %s", ds.upper(), missing)
            if found:
                df = pd.read_csv(path, usecols=found, low_memory=False)
                full_psg[ds] = df
                logger.info("  Loaded %d supplementary columns: %s", len(found), found)
            else:
                logger.warning("  No supplementary columns readable for %s.", ds.upper())
        except FileNotFoundError:
            logger.warning("  Full PSG file not found for %s: %s", ds.upper(), path)
        except Exception as exc:
            logger.error("  Failed to load full PSG columns for %s: %s", ds.upper(), exc)
    return full_psg


def col_available(ds, col, datasets, full_psg):
    """Return True if col exists for dataset ds in either harmonized or full PSG data."""
    if col is None:
        return False
    if ds in datasets and col in datasets[ds].columns:
        return True
    if ds in full_psg and col in full_psg[ds].columns:
        return True
    return False


# =============================================================================
# STEP 2 — Per-dataset profiling  [UNCHANGED]
# =============================================================================

def profile_datasets(datasets, logger):
    """
    Compute and log per-dataset profiling metrics.
    Handles boolean columns (e.g. TIHM snoring) separately from numeric stats.
    Saves descriptive statistics CSVs to OUTPUT_DIR.
    Returns {name: n_subjects}.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stat_filenames = {
        "shhs": "shhs_stats.csv",
        "mros": "mros_stats.csv",
        "mesa": "mesa_stats.csv",
        "tihm": "tihm_sleep_stats.csv",
    }
    subject_counts = {}

    for name, df in datasets.items():
        logger.info("=" * 60)
        logger.info("Profiling: %s", name.upper())
        logger.info("  Rows: %d  |  Columns: %d", *df.shape)

        # --- Unique subjects ---
        id_col = SUBJECT_ID.get(name)
        if id_col and id_col in df.columns:
            n_subjects = df[id_col].nunique()
            logger.info("  Unique subjects (%s): %d", id_col, n_subjects)
        else:
            n_subjects = df.shape[0]
            logger.info("  No subject ID column found; using row count as proxy.")
        subject_counts[name] = n_subjects

        # --- Missing values ---
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_summary = pd.concat([missing, missing_pct], axis=1, keys=["count", "pct"])
        cols_with_missing = missing_summary[missing_summary["count"] > 0]
        logger.info("  Columns with missing values: %d / %d", len(cols_with_missing), len(df.columns))
        if not cols_with_missing.empty:
            logger.info("  Missing value details:\n%s", cols_with_missing.to_string())

        # --- Boolean column stats (value counts + percentages) ---
        bool_cols = df.select_dtypes(include="bool").columns.tolist()
        for col in df.select_dtypes(include="object").columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({True, False, "True", "False", "true", "false"}):
                bool_cols.append(col)
        if bool_cols:
            logger.info("  Boolean columns (%d):", len(bool_cols))
            for col in bool_cols:
                vc  = df[col].value_counts(dropna=False)
                pct = (vc / len(df) * 100).round(2)
                detail = "  |  ".join(f"{k}: {v} ({pct[k]:.1f}%)" for k, v in vc.items())
                logger.info("    %s  →  %s", col, detail)

        # --- Numeric descriptive statistics ---
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            stats = numeric_df.describe().T
            logger.info("  Descriptive statistics (numeric columns):\n%s", stats.to_string())
            out_path = OUTPUT_DIR / stat_filenames[name]
            stats.to_csv(out_path)
            logger.info("  Stats saved → %s", out_path)

    return subject_counts


# =============================================================================
# STEP 3 — Feature alignment matrix (v3)
# =============================================================================

def build_alignment_matrix(datasets, full_psg, logger):
    """
    Build a string-valued presence matrix from FEATURE_MAP_V3.
    Values are "direct", "derivable", or "none".
    Also returns a boolean matrix for downstream PSG-vs-TIHM set analysis.
    """
    # --- v3 string matrix (for CSV) ---
    records_v3 = {}
    for feat, mapping in FEATURE_MAP_V3.items():
        records_v3[feat] = {ds: (val if val is not None else "none") for ds, val in mapping.items()}

    matrix_v3 = pd.DataFrame(records_v3).T
    matrix_v3.index.name = "feature"

    out_path = OUTPUT_DIR / "feature_alignment_matrix_v3.csv"
    matrix_v3.to_csv(out_path)
    logger.info("Feature alignment matrix (v3) saved → %s", out_path)

    # --- Boolean matrix for set logic (uses original FEATURE_MAP column checks) ---
    records_bool = {}
    for feat, mapping in FEATURE_MAP.items():
        records_bool[feat] = {
            ds: col_available(ds, col, datasets, full_psg)
            for ds, col in mapping.items()
        }
    matrix_bool = pd.DataFrame(records_bool).T
    matrix_bool.index.name = "feature"

    psg_cols = [d for d in PSG_DATASETS if d in matrix_bool.columns]
    tihm_col = "tihm"

    in_all_psg    = matrix_bool[psg_cols].all(axis=1) if psg_cols else pd.Series(False, index=matrix_bool.index)
    tihm_present  = matrix_bool[tihm_col] if tihm_col in matrix_bool.columns else pd.Series(False, index=matrix_bool.index)
    any_psg       = matrix_bool[psg_cols].any(axis=1) if psg_cols else pd.Series(False, index=matrix_bool.index)

    all_psg_features   = matrix_bool.index[in_all_psg].tolist()
    psg_only_features  = matrix_bool.index[any_psg & ~tihm_present].tolist()
    tihm_only_features = matrix_bool.index[tihm_present & ~any_psg].tolist()

    logger.info("Features in ALL 3 PSG datasets (%d): %s", len(all_psg_features), all_psg_features)
    logger.info("PSG-only features (%d): %s",              len(psg_only_features),  psg_only_features)
    logger.info("TIHM-only features (%d): %s",             len(tihm_only_features), tihm_only_features)

    return matrix_v3, all_psg_features, psg_only_features, tihm_only_features


# =============================================================================
# STEP 4 — Visualisations
# =============================================================================

# ---------------------------------------------------------------------------
# Plot 02 — Feature overlap heatmap (3-color, category-grouped)
# ---------------------------------------------------------------------------

def plot_feature_overlap_heatmap(logger):
    """
    Plot 02: category-grouped heatmap using FEATURE_MAP_V3.
    Green = direct, amber = derivable, light gray = not available.
    """
    ds_order       = ["shhs", "mros", "mesa", "tihm"]
    val_to_int     = {None: 0, "derivable": 1, "direct": 2}
    cell_colors    = ["#EEEEEE", "#FF9800", "#4CAF50"]
    cell_cmap      = mpl.colors.ListedColormap(cell_colors)

    # Build ordered feature list from categories
    ordered_feats = [f for feats in FEATURE_CATEGORIES.values() for f in feats]

    # Build integer matrix
    data = np.zeros((len(ordered_feats), len(ds_order)), dtype=int)
    for i, feat in enumerate(ordered_feats):
        for j, ds in enumerate(ds_order):
            val = FEATURE_MAP_V3.get(feat, {}).get(ds)
            data[i, j] = val_to_int.get(val, 0)

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(data, cmap=cell_cmap, vmin=0, vmax=2, aspect="auto")

    # --- Grid lines between cells ---
    for x in np.arange(-0.5, len(ds_order), 1):
        ax.axvline(x, color="white", linewidth=1.5)
    for y in np.arange(-0.5, len(ordered_feats), 1):
        ax.axhline(y, color="white", linewidth=0.8)

    # --- Category separator lines and labels ---
    cumulative = 0
    for cat, feats in FEATURE_CATEGORIES.items():
        if cumulative > 0:
            ax.axhline(cumulative - 0.5, color="white", linewidth=3, zorder=5)
        mid_y = cumulative + (len(feats) - 1) / 2
        ax.text(-0.65, mid_y, cat,
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color="#444444",
                transform=ax.transData, clip_on=False)
        cumulative += len(feats)

    # --- Axes labels ---
    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels([d.upper() for d in ds_order], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(ordered_feats)))
    ax.set_yticklabels(ordered_feats, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="both", length=0)

    # --- Legend ---
    legend_handles = [
        mpatches.Patch(facecolor="#4CAF50", label="Direct"),
        mpatches.Patch(facecolor="#FF9800", label="Derivable"),
        mpatches.Patch(facecolor="#EEEEEE", edgecolor="#CCCCCC", label="Not available"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, -0.04), ncol=3, fontsize=10,
              framealpha=0.95, edgecolor="#CCCCCC")

    ax.set_title("Feature Availability Across Datasets", fontsize=14, fontweight="bold", pad=14)
    ax.spines[:].set_visible(False)
    ax.grid(False)

    plt.subplots_adjust(left=0.28)
    save_figure(fig, OUTPUT_DIR, "02_feature_overlap_heatmap.svg")
    logger.info("Plot saved → 02_feature_overlap_heatmap.svg")


# ---------------------------------------------------------------------------
# Plot 03 — Shared feature distributions (PSG KDE overlay)
# ---------------------------------------------------------------------------

def plot_shared_distributions(datasets, logger):
    """Plot 03: KDE overlay of shared PSG features across datasets."""
    n_cols = 3
    n_rows = int(np.ceil(len(DIST_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()

    for idx, feat in enumerate(DIST_FEATURES):
        ax = axes[idx]
        mapping = FEATURE_MAP.get(feat, {})
        plotted = False
        for ds in PSG_DATASETS:
            col = mapping.get(ds)
            if col and ds in datasets and col in datasets[ds].columns:
                series = pd.to_numeric(datasets[ds][col], errors="coerce").dropna()
                if len(series) > 1:
                    sns.kdeplot(series, ax=ax, label=ds.upper(),
                                color=DATASET_COLORS[ds], linewidth=2)
                    plotted = True
        ax.set_title(feat.replace("_", " ").title())
        ax.set_ylabel("Density")
        if plotted:
            ax.legend()

    for ax in axes[len(DIST_FEATURES):]:
        ax.set_visible(False)

    fig.suptitle("Shared Feature Distributions — PSG Datasets", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "03_shared_feature_distributions.svg")
    logger.info("Plot saved → 03_shared_feature_distributions.svg")


# ---------------------------------------------------------------------------
# Plot 04 — Correlation heatmaps (one per PSG dataset)
# ---------------------------------------------------------------------------

def plot_correlation_heatmaps(datasets, logger):
    """Plot 04: correlation heatmap for each PSG harmonized dataset."""
    ds_full_names = {"shhs": "SHHS", "mros": "MrOS", "mesa": "MESA"}
    for ds in PSG_DATASETS:
        if ds not in datasets:
            logger.warning("Skipping correlation heatmap for %s — not loaded.", ds)
            continue
        numeric_df = datasets[ds].select_dtypes(include="number")
        n_cols = numeric_df.shape[1]
        if n_cols < 2:
            logger.warning("Skipping correlation heatmap for %s — fewer than 2 numeric columns.", ds)
            continue

        corr     = numeric_df.corr()
        annotate = n_cols <= 25

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    annot=annotate, fmt=".2f" if annotate else "",
                    linewidths=0.3 if annotate else 0,
                    ax=ax)
        ax.set_title(
            f"{ds_full_names[ds]} Harmonized Dataset — Pearson Correlation Matrix\n"
            f"({n_cols} numeric features)",
            fontweight="bold",
        )
        plt.tight_layout()
        fname = f"04_correlation_heatmap_{ds}.svg"
        save_figure(fig, OUTPUT_DIR, fname)
        logger.info("Plot saved → %s", fname)


# ---------------------------------------------------------------------------
# Plot 05 — TIHM feature distributions (2×2 grid)
# ---------------------------------------------------------------------------

def plot_tihm_distributions(datasets, logger):
    """Plot 05: 2×2 distribution grid for TIHM Sleep features."""
    if "tihm" not in datasets:
        logger.warning("TIHM data not loaded — skipping distribution plot.")
        return

    df    = datasets["tihm"]
    color = DATASET_COLORS["tihm"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: heart_rate
    ax = axes[0, 0]
    if "heart_rate" in df.columns:
        sns.histplot(df["heart_rate"].dropna(), kde=True, ax=ax, color=color)
        ax.set_title("Heart Rate")
        ax.set_xlabel("BPM")
    else:
        ax.set_visible(False)

    # Top-right: respiratory_rate
    ax = axes[0, 1]
    if "respiratory_rate" in df.columns:
        sns.histplot(df["respiratory_rate"].dropna(), kde=True, ax=ax, color=color)
        ax.set_title("Respiratory Rate")
        ax.set_xlabel("Breaths / min")
    else:
        ax.set_visible(False)

    # Bottom-left: snoring — boolean value counts with % labels
    ax = axes[1, 0]
    if "snoring" in df.columns:
        vc   = df["snoring"].value_counts(dropna=False)
        pct  = (vc / len(df) * 100).round(1)
        bars = ax.bar([str(k) for k in vc.index], vc.values,
                      color=color, edgecolor="white")
        for bar, (k, p) in zip(bars, pct.items()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + len(df) * 0.005,
                    f"{p}%", ha="center", va="bottom", fontsize=9)
        ax.set_title("Snoring (True / False)")
        ax.set_ylabel("Count")
    else:
        ax.set_visible(False)

    # Bottom-right: sleep/wake state
    ax = axes[1, 1]
    if "state" in df.columns:
        vc = df["state"].value_counts()
        vc.plot(kind="bar", ax=ax, color=color, edgecolor="white")
        ax.set_title("Sleep / Wake State")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.set_visible(False)

    fig.suptitle("TIHM Sleep — Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "05_tihm_distributions.svg")
    logger.info("Plot saved → 05_tihm_distributions.svg")


# ---------------------------------------------------------------------------
# Plot 06 — TIHM longitudinal time series (dual-axis)
# ---------------------------------------------------------------------------

def plot_tihm_over_time(datasets, logger):
    """
    Plot 06: dual-axis HR + RR line chart for the TIHM patient with the most records.
    Background coloured green (sleep) / orange (awake).
    """
    if "tihm" not in datasets:
        logger.warning("TIHM data not loaded — skipping time series plot.")
        return

    df = datasets["tihm"].copy()
    required = {"date", "patient_id", "heart_rate", "respiratory_rate"}
    if not required.issubset(df.columns):
        logger.warning("TIHM missing columns for time series plot: %s", required - set(df.columns))
        return

    top_patient = df["patient_id"].value_counts().index[0]
    pat_df = df[df["patient_id"] == top_patient].sort_values("date").copy()
    logger.info("Time series plot: patient %s (%d records)", top_patient, len(pat_df))

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # --- Background colouring by sleep/wake state ---
    if "state" in pat_df.columns:
        unique_states = pat_df["state"].dropna().unique()
        logger.info("  Unique state values: %s", unique_states.tolist())

        def is_sleep(val):
            return str(val).strip().upper() in {"SLEEP", "S", "1"}

        state_color = {s: ("#4CAF50" if is_sleep(s) else "#FF9800") for s in unique_states}
        pat_df["_run"] = (pat_df["state"] != pat_df["state"].shift()).cumsum()
        for _, grp in pat_df.groupby("_run", sort=False):
            s_val = grp["state"].iloc[0]
            ax1.axvspan(grp["date"].iloc[0], grp["date"].iloc[-1],
                        alpha=0.25, color=state_color.get(s_val, "#E0E0E0"), linewidth=0)

        sleep_patch = mpatches.Patch(color="#4CAF50", alpha=0.5, label="Sleep")
        awake_patch = mpatches.Patch(color="#FF9800", alpha=0.5, label="Awake")

    # --- Heart rate — left axis (blue) ---
    hr = pd.to_numeric(pat_df["heart_rate"], errors="coerce")
    ax1.plot(pat_df["date"], hr, color="#2196F3", linewidth=1,
             label="Heart Rate (BPM)", alpha=0.9, zorder=3)
    ax1.set_ylabel("Heart Rate (BPM)", color="#2196F3", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax1.set_xlabel("Time")

    # --- Respiratory rate — right axis (purple) ---
    ax2 = ax1.twinx()
    rr = pd.to_numeric(pat_df["respiratory_rate"], errors="coerce")
    ax2.plot(pat_df["date"], rr, color="#9C27B0", linewidth=1,
             label="Resp. Rate (breaths/min)", alpha=0.9, zorder=3)
    ax2.set_ylabel("Respiratory Rate (breaths/min)", color="#9C27B0", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#9C27B0")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#CCCCCC")

    ax1.set_title(
        f"TIHM Patient {top_patient} — Heart Rate & Respiratory Rate Over Time\n"
        "(background: green = sleep, orange = awake)",
        fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    state_handles   = [sleep_patch, awake_patch] if "state" in pat_df.columns else []
    ax1.legend(lines1 + lines2 + state_handles,
               labels1 + labels2 + [p.get_label() for p in state_handles],
               loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "06_tihm_sleep_over_time.svg")
    logger.info("Plot saved → 06_tihm_sleep_over_time.svg")


# ---------------------------------------------------------------------------
# Plot 07 — Aligned feature summary table
# ---------------------------------------------------------------------------

def plot_aligned_feature_summary(logger):
    """
    Plot 07: matplotlib table showing features available in TIHM AND at least one PSG dataset.
    Columns: Feature | Category | SHHS | MrOS | MESA | TIHM | Notes
    """
    # Collect rows: tihm != None AND at least one PSG has a value
    rows_data = []
    for feat, mapping in FEATURE_MAP_V3.items():
        tihm_val = mapping.get("tihm")
        psg_vals = [mapping.get(ds) for ds in PSG_DATASETS]
        if tihm_val is None:
            continue
        if all(v is None for v in psg_vals):
            continue

        def fmt(v):
            if v == "direct":     return "Direct"
            if v == "derivable":  return "Derivable"
            return "-"

        note = TIHM_DERIVATION_NOTES.get(feat, "")
        rows_data.append([
            feat,
            FEATURE_TO_CATEGORY.get(feat, ""),
            fmt(mapping.get("shhs")),
            fmt(mapping.get("mros")),
            fmt(mapping.get("mesa")),
            fmt(tihm_val),
            note,
        ])

    headers    = ["Feature", "Category", "SHHS", "MrOS", "MESA", "TIHM", "Notes"]
    col_widths = [0.14, 0.16, 0.07, 0.07, 0.07, 0.09, 0.40]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows_data,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    n_rows = len(rows_data)
    n_cols = len(headers)

    # Header row
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#37474F")

    # Data rows — alternating background, colored status cells
    status_colors = {"Direct": "#E3F2FD", "Derivable": "#FFF3E0", "-": "#F5F5F5"}
    for row_idx in range(1, n_rows + 1):
        row_bg = "white" if row_idx % 2 == 1 else "#F5F5F5"
        for col_idx in range(n_cols):
            cell = tbl[row_idx, col_idx]
            cell.set_edgecolor("#E0E0E0")
            val = rows_data[row_idx - 1][col_idx]
            if col_idx in (2, 3, 4, 5) and val in status_colors:
                cell.set_facecolor(status_colors[val])
            else:
                cell.set_facecolor(row_bg)

    ax.set_title("Final Aligned Feature Set — PSG and TIHM",
                 fontsize=13, fontweight="bold", pad=16, y=0.97)

    # Legend below table
    legend_handles = [
        mpatches.Patch(facecolor="#E3F2FD", edgecolor="#CCCCCC", label="Direct"),
        mpatches.Patch(facecolor="#FFF3E0", edgecolor="#CCCCCC", label="Derivable"),
        mpatches.Patch(facecolor="#F5F5F5", edgecolor="#CCCCCC", label="Not available"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=9,
              framealpha=0.95, edgecolor="#CCCCCC")

    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "07_aligned_feature_summary.svg")
    logger.info("Plot saved → 07_aligned_feature_summary.svg")


# =============================================================================
# STEP 5 — Granularity Analysis  [UNCHANGED]
# =============================================================================

def log_granularity_analysis(datasets, logger):
    """Log data granularity (time resolution) across datasets and alignment implications."""
    logger.info("=" * 60)
    logger.info("GRANULARITY ANALYSIS")
    logger.info("=" * 60)

    granularity = {
        "shhs": {
            "resolution":   "Per-night summary",
            "rows_meaning": "One row = one sleep study night per subject",
            "file_used":    "Harmonized dataset",
            "note":         "Full PSG file contains per-night derived metrics (HR avg/min/max over full study)",
        },
        "mros": {
            "resolution":   "Per-night summary",
            "rows_meaning": "One row = one visit night per subject",
            "file_used":    "Harmonized dataset",
            "note":         "Visit 1 and Visit 2 are separate files; each is a per-night summary",
        },
        "mesa": {
            "resolution":   "Per-night summary",
            "rows_meaning": "One row = one sleep study night per subject",
            "file_used":    "Harmonized dataset",
            "note":         "Also includes actigraphy-derived averages over ~7 nights of wrist actigraphy",
        },
        "tihm": {
            "resolution":   "Per-minute epoch (longitudinal daily)",
            "rows_meaning": "One row = one 1-minute epoch; multiple nights per patient",
            "file_used":    "Sleep.csv",
            "note":         "Wearable IoT device — continuous multi-night recording with state labels per epoch",
        },
    }

    for ds, info in granularity.items():
        if ds not in datasets:
            continue
        df = datasets[ds]
        logger.info("")
        logger.info("  %s:", ds.upper())
        logger.info("    Resolution   : %s", info["resolution"])
        logger.info("    Row meaning  : %s", info["rows_meaning"])
        logger.info("    Total rows   : %d", len(df))
        logger.info("    File used    : %s", info["file_used"])
        logger.info("    Note         : %s", info["note"])

    logger.info("")
    logger.info("  KEY FINDING:")
    logger.info("    PSG harmonized files are PER-NIGHT SUMMARIES (1 row per subject per study night).")
    logger.info("    TIHM Sleep.csv is PER-MINUTE EPOCH DATA (many rows per patient across many nights).")
    logger.info("")
    logger.info("  IMPLICATION:")
    logger.info("    Direct column comparison is NOT valid without alignment of granularity.")
    logger.info("    Option A: Aggregate TIHM epochs to per-night summaries (mean HR, RR; sleep fraction).")
    logger.info("    Option B: Extract epoch-level features from full PSG signal files (not used here).")
    logger.info("")
    logger.info("  RECOMMENDATION:")
    logger.info("    For Stage 02 (PSG-wearable subset), use Option A:")
    logger.info("    Aggregate TIHM Sleep.csv per patient_id + night to produce per-night HR avg/min/max,")
    logger.info("    sleep efficiency (pct epochs labelled SLEEP), and snoring rate.")
    logger.info("    These can then be directly compared against the PSG harmonized per-night summaries.")
    logger.info("=" * 60)


# =============================================================================
# STEP 6 — Summary log  [UNCHANGED]
# =============================================================================

def log_summary(datasets, subject_counts, all_psg_features,
                psg_only_features, tihm_only_features, output_files, logger):
    """Print and log a structured final summary."""
    lines = [
        "",
        "=" * 60,
        "  SUMMARY",
        "=" * 60,
        f"  Datasets loaded : {len(datasets)} / 4",
        "",
        f"  {'Dataset':<10} {'Rows':>8}  {'Subjects':>10}",
        f"  {'-'*10}  {'-'*8}  {'-'*10}",
    ]
    for name, df in datasets.items():
        lines.append(f"  {name.upper():<10} {df.shape[0]:>8,}  {subject_counts[name]:>10,}")

    lines += [
        "",
        f"  Total features in alignment map : {len(FEATURE_MAP_V3)}",
        f"  In all 3 PSG datasets           : {len(all_psg_features)}",
        f"  PSG-only features               : {len(psg_only_features)}",
        f"  TIHM-only features              : {len(tihm_only_features)}",
        "",
        "  Output files generated:",
    ]
    for f in output_files:
        lines.append(f"    {f}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print(summary)
    logger.info(summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = get_logger("01_dataset_overview", "all")
    logger.info("Starting Stage 01: Dataset Overview")

    # Apply uniform thesis style before any plots are generated
    apply_thesis_style()

    # --- Step 1: Load ---
    datasets = load_datasets(logger)
    if not datasets:
        logger.error("No datasets loaded. Check file paths at the top of this script.")
        sys.exit(1)

    full_psg = load_full_psg_columns(logger)

    # --- Step 2: Profile ---
    subject_counts = profile_datasets(datasets, logger)

    # --- Step 3: Feature alignment matrix (v3) ---
    matrix_v3, all_psg_features, psg_only_features, tihm_only_features = \
        build_alignment_matrix(datasets, full_psg, logger)

    # --- Step 4: Plots ---
    plot_feature_overlap_heatmap(logger)                    # 02
    plot_shared_distributions(datasets, logger)             # 03
    plot_correlation_heatmaps(datasets, logger)             # 04
    plot_tihm_distributions(datasets, logger)               # 05
    plot_tihm_over_time(datasets, logger)                   # 06
    plot_aligned_feature_summary(logger)                    # 07

    # --- Step 5: Granularity analysis ---
    log_granularity_analysis(datasets, logger)

    # --- Step 6: Final summary ---
    output_files = [
        "outputs/overview/shhs_stats.csv",
        "outputs/overview/mros_stats.csv",
        "outputs/overview/mesa_stats.csv",
        "outputs/overview/tihm_sleep_stats.csv",
        "outputs/overview/feature_alignment_matrix_v3.csv",
        "outputs/overview/02_feature_overlap_heatmap.svg",
        "outputs/overview/03_shared_feature_distributions.svg",
        "outputs/overview/04_correlation_heatmap_shhs.svg",
        "outputs/overview/04_correlation_heatmap_mros.svg",
        "outputs/overview/04_correlation_heatmap_mesa.svg",
        "outputs/overview/05_tihm_distributions.svg",
        "outputs/overview/06_tihm_sleep_over_time.svg",
        "outputs/overview/07_aligned_feature_summary.svg",
    ]
    log_summary(datasets, subject_counts, all_psg_features,
                psg_only_features, tihm_only_features, output_files, logger)

    logger.info("Stage 01 complete. All outputs in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
