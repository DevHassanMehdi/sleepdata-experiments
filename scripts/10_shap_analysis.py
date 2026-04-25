"""
scripts/10_shap_analysis.py
============================
SHAP-based feature importance and interpretability analysis.

Runs on Step 2 (MESA in-distribution) and Step 3 (cross-dataset TIHM) models
to explain which features drive sleep-stage predictions and how feature
importance shifts between clinical PSG and wearable data.

Focuses on RF and XGBoost — TreeExplainer is fast and exact for these.
MLP/LSTM require KernelExplainer (prohibitively slow) and are excluded.

Usage:
  python scripts/10_shap_analysis.py --model xgboost --step 2
  python scripts/10_shap_analysis.py --model random_forest --step 3
  python scripts/10_shap_analysis.py --all
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    print("[warn] shap not installed. Run: pip install shap", flush=True)


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "seed": 42,
    "label_names": ["AWAKE", "LIGHT", "DEEP", "REM"],
    "n_shap_samples": 2000,

    "paths": {
        "mesa_aligned": "outputs/features/aligned/",
        "tihm_aligned": "outputs/features/tihm_aligned_all.csv",
        "models":       "outputs/models/",
        "results":      "outputs/results/shap/",
        "figures":      "outputs/figures/shap/",
        "logs":         "logs/shap/",
    },
}

FEATURE_COLS = [
    "hr_mean", "hr_median", "rr_mean", "snore_pct",
    "hr_lag1", "hr_lag2", "hr_lag3",
    "rr_lag1", "rr_lag2", "rr_lag3", "snore_lag1",
    "hr_rolling_mean_5", "hr_rolling_std_5",
    "rr_rolling_mean_5", "rr_rolling_std_5",
    "age_group", "sex",
]
assert len(FEATURE_COLS) == 17

_MODEL_ABBR = {"random_forest": "rf", "xgboost": "xgb"}

# Colour per feature group for importance bar charts
_FEATURE_COLORS = {
    "hr_mean":            "#1f77b4",
    "hr_median":          "#1f77b4",
    "hr_rolling_mean_5":  "#1f77b4",
    "hr_rolling_std_5":   "#1f77b4",
    "hr_lag1":            "#9467bd",
    "hr_lag2":            "#9467bd",
    "hr_lag3":            "#9467bd",
    "rr_mean":            "#2ca02c",
    "rr_rolling_mean_5":  "#2ca02c",
    "rr_rolling_std_5":   "#2ca02c",
    "rr_lag1":            "#9467bd",
    "rr_lag2":            "#9467bd",
    "rr_lag3":            "#9467bd",
    "snore_pct":          "#ff7f0e",
    "snore_lag1":         "#ff7f0e",
    "age_group":          "#7f7f7f",
    "sex":                "#7f7f7f",
}


# =============================================================================
# Helpers
# =============================================================================

def _p(key: str) -> Path:
    return ROOT_DIR / CONFIG["paths"][key]


def _encode_labels(series: pd.Series) -> pd.Series:
    label_map = {"AWAKE": 0, "LIGHT": 1, "DEEP": 2, "REM": 3}
    if not pd.api.types.is_integer_dtype(series):
        return series.map(label_map)
    return series


# =============================================================================
# Data loading  (X returned as DataFrame so SHAP plots show feature names)
# =============================================================================

def load_mesa_aligned() -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X DataFrame, y ndarray) from all MESA aligned CSVs."""
    aligned_dir = _p("mesa_aligned")
    files       = sorted(aligned_dir.glob("mesa_aligned_*.csv"))
    if not files:
        raise FileNotFoundError(f"No MESA aligned CSVs in {aligned_dir}")

    frames = []
    for f in files:
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df["label"] = _encode_labels(df["label"])
    df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    X = df[FEATURE_COLS].copy()
    y = df["label"].values
    print(f"  MESA aligned : {len(X):,} epochs, {len(files)} subjects",
          flush=True)
    return X, y


def load_tihm_aligned() -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X DataFrame, y ndarray) from TIHM aligned CSV."""
    tihm_path = ROOT_DIR / CONFIG["paths"]["tihm_aligned"]
    if not tihm_path.exists():
        raise FileNotFoundError(f"TIHM aligned not found: {tihm_path}")
    df = pd.read_csv(tihm_path)
    df["label"] = _encode_labels(df["label"])
    df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    X = df[FEATURE_COLS].copy()
    y = df["label"].values
    print(f"  TIHM aligned : {len(X):,} epochs", flush=True)
    return X, y


# =============================================================================
# SHAP computation
# =============================================================================

def compute_shap_tree(model, X_explain: pd.DataFrame,
                      model_name: str, step: int) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer on a random subsample.

    Returns shap_values of shape (n_samples, n_features, n_classes).
    """
    if not _SHAP_OK:
        raise ImportError("shap not installed. Run: pip install shap")

    n = min(CONFIG["n_shap_samples"], len(X_explain))
    rng     = np.random.default_rng(CONFIG["seed"])
    idx     = rng.choice(len(X_explain), n, replace=False)
    X_sub   = X_explain.iloc[idx].reset_index(drop=True)

    print(f"  Computing SHAP values on {n:,} samples ...", flush=True)
    t0 = time.time()

    explainer   = shap.TreeExplainer(model)
    shap_output = explainer.shap_values(X_sub)

    # sklearn RF returns list[array] (one per class); XGBoost returns (n,f,c)
    if isinstance(shap_output, list):
        # Stack list of (n, f) arrays → (n, f, c)
        shap_values = np.stack(shap_output, axis=2)
    else:
        shap_values = shap_output

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)
    return shap_values, X_sub


# =============================================================================
# Plots
# =============================================================================

def plot_shap_importance(shap_values: np.ndarray, feature_names: list,
                         model_name: str, step: int) -> Path:
    """
    Global feature importance: mean |SHAP| averaged across all classes.
    """
    _p("figures").mkdir(parents=True, exist_ok=True)
    out_path = _p("figures") / f"importance_step{step}_{_MODEL_ABBR[model_name]}.svg"

    # Mean |SHAP| per feature across all classes and samples: (n,f,c) → (f,)
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    order    = np.argsort(mean_abs)          # ascending for barh

    colors = [_FEATURE_COLORS.get(feature_names[i], "#333333") for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order)), mean_abs[order], color=colors, alpha=0.85)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance — {model_name.replace('_', ' ').title()} "
                 f"Step {step}")
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def plot_shap_per_class(shap_values: np.ndarray, feature_names: list,
                        model_name: str, step: int) -> Path:
    """
    Per-class feature importance: top-10 features for each sleep stage.
    """
    _p("figures").mkdir(parents=True, exist_ok=True)
    out_path = _p("figures") / f"per_class_step{step}_{_MODEL_ABBR[model_name]}.svg"

    label_names = CONFIG["label_names"]
    n_classes   = shap_values.shape[2]
    top_n       = min(10, len(feature_names))

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 5),
                             sharey=False)
    if n_classes == 1:
        axes = [axes]

    for cls_idx, ax in enumerate(axes):
        cls_name    = label_names[cls_idx] if cls_idx < len(label_names) else str(cls_idx)
        cls_shap    = np.abs(shap_values[:, :, cls_idx]).mean(axis=0)  # (f,)
        top_idx     = np.argsort(cls_shap)[::-1][:top_n]
        top_vals    = cls_shap[top_idx]
        top_names   = [feature_names[i] for i in top_idx]
        colors      = [_FEATURE_COLORS.get(feature_names[i], "#333333")
                       for i in top_idx]

        ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=7)
        ax.set_xlabel("Mean |SHAP|", fontsize=8)
        ax.set_title(cls_name, fontsize=9)

    fig.suptitle(f"Per-class SHAP — {model_name.replace('_', ' ').title()} "
                 f"Step {step}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def plot_shap_beeswarm(shap_values: np.ndarray, X_sample: pd.DataFrame,
                       feature_names: list, model_name: str,
                       step: int) -> list[Path]:
    """
    SHAP beeswarm (summary) plot — one SVG per class.
    """
    if not _SHAP_OK:
        return []

    _p("figures").mkdir(parents=True, exist_ok=True)
    label_names = CONFIG["label_names"]
    saved_paths = []

    for cls_idx in range(shap_values.shape[2]):
        cls_name = (label_names[cls_idx]
                    if cls_idx < len(label_names) else str(cls_idx))
        out_path = (_p("figures") /
                    f"beeswarm_step{step}_{_MODEL_ABBR[model_name]}_{cls_name}.svg")

        fig = plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_values[:, :, cls_idx],
            X_sample,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=17,
        )
        plt.title(f"{model_name.replace('_', ' ').title()} Step {step} — {cls_name}",
                  fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def plot_step2_vs_step3_comparison(shap_step2: np.ndarray,
                                   shap_step3: np.ndarray,
                                   feature_names: list,
                                   model_name: str) -> Path:
    """
    Side-by-side bar chart comparing Step 2 (MESA) vs Step 3 (TIHM) importance.
    """
    _p("figures").mkdir(parents=True, exist_ok=True)
    out_path = (_p("figures") /
                f"comparison_step2_vs_step3_{_MODEL_ABBR[model_name]}.svg")

    imp_step2 = np.abs(shap_step2).mean(axis=(0, 2))
    imp_step3 = np.abs(shap_step3).mean(axis=(0, 2))

    # Sort by mean of both importances
    order = np.argsort((imp_step2 + imp_step3) / 2)

    y      = np.arange(len(feature_names))
    height = 0.35
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(y - height / 2, imp_step2[order], height,
            label="Step 2 (MESA test)",  color="#1f77b4", alpha=0.8)
    ax.barh(y + height / 2, imp_step3[order], height,
            label="Step 3 (TIHM test)", color="#ff7f0e", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Step 2 vs Step 3 Feature Importance — "
                 f"{model_name.replace('_', ' ').title()}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


# =============================================================================
# Results CSV
# =============================================================================

def save_shap_csv(shap_values: np.ndarray, feature_names: list,
                  model_name: str, step: int) -> Path:
    """Save per-feature mean |SHAP| values to CSV."""
    _p("results").mkdir(parents=True, exist_ok=True)
    out_path = (_p("results") /
                f"shap_importance_step{step}_{_MODEL_ABBR[model_name]}.csv")

    label_names = CONFIG["label_names"]
    rows = []
    for fi, feat in enumerate(feature_names):
        row = {"feature": feat,
               "mean_abs_shap": float(np.abs(shap_values[:, fi, :]).mean())}
        for ci, cls in enumerate(label_names):
            row[f"shap_{cls.lower()}"] = float(
                np.abs(shap_values[:, fi, ci]).mean()
            )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    df.to_csv(out_path, index=False)
    return out_path


# =============================================================================
# Per-combination entry point
# =============================================================================

def run_shap(model_name: str, step: int) -> np.ndarray:
    """
    Run full SHAP pipeline for one (model_name, step) combination.
    Returns shap_values for use in comparison plots.
    """
    abbr = _MODEL_ABBR[model_name]
    SEP  = "═" * 42
    print(f"\n{SEP}", flush=True)
    print(f" SHAP Analysis", flush=True)
    print(f" Model: {model_name.replace('_', ' ').title()}  Step: {step}", flush=True)
    print(SEP, flush=True)

    # Load model
    model_path = _p("models") / f"step{step}_{abbr}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run training first:  python scripts/07_train_models.py "
            f"--step {step} --model {model_name}"
        )
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    # Load data — Step 2 uses MESA, Step 3 uses TIHM for explanation set
    print("\n Loading data ...", flush=True)
    if step == 2:
        X, y = load_mesa_aligned()
    elif step == 3:
        X, _ = load_mesa_aligned()    # MESA for background (unused by TreeExplainer)
        X, y = load_tihm_aligned()    # TIHM for explanation
    else:
        raise ValueError(f"Unsupported step: {step} — choose 2 or 3")

    # Compute SHAP
    t_total = time.time()
    shap_values, X_sample = compute_shap_tree(model, X, model_name, step)

    # Print top-5 features
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    top5     = np.argsort(mean_abs)[::-1][:5]
    print(f"\n Top 5 features by mean |SHAP|:", flush=True)
    for rank, fi in enumerate(top5, 1):
        print(f"   {rank}. {FEATURE_COLS[fi]:<25} {mean_abs[fi]:.4f}", flush=True)

    # Plots
    print(flush=True)
    imp_path  = plot_shap_importance(shap_values, FEATURE_COLS, model_name, step)
    cls_path  = plot_shap_per_class(shap_values, FEATURE_COLS, model_name, step)
    bee_paths = plot_shap_beeswarm(shap_values, X_sample, FEATURE_COLS,
                                   model_name, step)
    csv_path  = save_shap_csv(shap_values, FEATURE_COLS, model_name, step)

    print(f" Saved: {imp_path.relative_to(ROOT_DIR)}", flush=True)
    print(f" Saved: {cls_path.relative_to(ROOT_DIR)}", flush=True)
    for bp in bee_paths:
        print(f" Saved: {bp.relative_to(ROOT_DIR)}", flush=True)
    print(f" Saved: {csv_path.relative_to(ROOT_DIR)}", flush=True)
    print(SEP, flush=True)

    return shap_values


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="python scripts/10_shap_analysis.py",
        description="SHAP feature importance analysis for sleep staging models.",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true",
                     help="Run all four combinations and generate comparison plots.")
    grp.add_argument("--model", type=str,
                     choices=["random_forest", "xgboost"],
                     help="Model to analyse.")
    parser.add_argument("--step", type=int, choices=[2, 3],
                        help="Step (2=MESA CV, 3=cross-dataset TIHM). "
                             "Required unless --all.")
    args = parser.parse_args()

    if not args.all and args.step is None:
        parser.error("--step is required when --model is specified.")

    if not _SHAP_OK:
        print("ERROR: shap not installed. Run: pip install shap", flush=True)
        sys.exit(1)

    stored: dict[str, dict[int, np.ndarray]] = {}

    if args.all:
        for model_name in ("xgboost", "random_forest"):
            stored[model_name] = {}
            for step in (2, 3):
                shap_vals = run_shap(model_name, step)
                stored[model_name][step] = shap_vals

        # Comparison plots (Step 2 vs Step 3) for each model
        print("\nGenerating Step 2 vs Step 3 comparison plots ...", flush=True)
        for model_name, step_vals in stored.items():
            if 2 in step_vals and 3 in step_vals:
                out = plot_step2_vs_step3_comparison(
                    step_vals[2], step_vals[3], FEATURE_COLS, model_name
                )
                print(f" Saved: {out.relative_to(ROOT_DIR)}", flush=True)
    else:
        run_shap(args.model, args.step)


if __name__ == "__main__":
    main()
