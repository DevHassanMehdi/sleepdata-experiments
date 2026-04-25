"""
scripts/07_train_models.py
==========================
ML training pipeline for sleep-stage classification experiments.

Steps:
  1 — Within-dataset 5-fold GroupKFold CV on MESA full PSG features (2184 feats)
  2 — Within-dataset 5-fold GroupKFold CV on MESA aligned features (17 feats)
  3 — Cross-dataset: train on MESA aligned → evaluate on TIHM aligned
  4 — Cross-dataset: train on MESA harmonized → evaluate on TIHM harmonized
      (requires script 08 to have been run first)

Usage:
  python scripts/07_train_models.py --step 1 --model random_forest
  python scripts/07_train_models.py --step 2 --model xgboost
  python scripts/07_train_models.py --step 3
  python scripts/07_train_models.py --step 4
  python scripts/07_train_models.py --all
  python scripts/07_train_models.py --all --resume
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight as _sklearn_cw

try:
    from xgboost import XGBClassifier
    _XGBOOST_OK = True
except ImportError:
    _XGBOOST_OK = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    _TORCH_OK = True
    _TB_OK    = True
    _DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH_OK = False
    _TB_OK    = False
    _DEVICE   = None

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# PyTorch utils (imported lazily so the script works without torch installed)
if _TORCH_OK:
    from utils.data_utils     import EpochDataset, SequenceDataset
    from utils.training_utils import train_pytorch_model
    from utils.models.mlp     import SleepMLP
    from utils.models.lstm    import SleepLSTM
    from utils.models.cnn     import SleepCNN


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "seed": 42,
    "n_folds": 5,
    "label_map": {"AWAKE": 0, "LIGHT": 1, "DEEP": 2, "REM": 3},
    "label_names": ["AWAKE", "LIGHT", "DEEP", "REM"],

    "paths": {
        "mesa_full":    "outputs/features/full/",
        "mesa_aligned": "outputs/features/aligned/",
        "tihm_aligned": "outputs/features/tihm_aligned_all.csv",
        "tihm_full":    "outputs/features/tihm_full_all.csv",
        "results":      "outputs/results/",
        "predictions":  "outputs/predictions/",
        "models":       "outputs/models/",
        "figures":      "outputs/figures/",
        "tensorboard":  "outputs/tensorboard/",
        "logs":         "logs/training/",
    },

    "random_forest": {
        "n_estimators":    500,
        "max_features":    "sqrt",
        "min_samples_leaf": 5,
        "n_jobs":          -1,
        "class_weight":    "balanced",
        "random_state":    42,
    },

    "xgboost": {
        "n_estimators":    500,
        "max_depth":       6,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "eval_metric":     "mlogloss",
        "random_state":    42,
        "n_jobs":          -1,
    },

    "feature_selection": {
        "enabled":          True,
        "n_features":       200,
        "quick_estimators": 100,
    },

    "mlp": {
        "hidden_dims":   [256, 128, 64],
        "dropout":       0.3,
        "learning_rate": 0.001,
        "batch_size":    512,
        "epochs":        100,
        "patience":      10,
        "weight_decay":  1e-4,
    },

    "lstm": {
        "hidden_size":   128,
        "num_layers":    2,
        "dropout":       0.3,
        "bidirectional": True,
        "seq_len":       10,
        "learning_rate": 0.001,
        "batch_size":    256,
        "epochs":        100,
        "patience":      10,
        "weight_decay":  1e-4,
    },

    "cnn": {
        "dropout":       0.3,
        "learning_rate": 0.001,
        "batch_size":    512,
        "epochs":        100,
        "patience":      10,
        "weight_decay":  1e-4,
    },

    "device": "auto",   # "auto" → cuda if available, else cpu
}

# Columns that are metadata, not features
_NON_FEATURE = {"label", "patient_id", "session_id", "epoch_index"}

_MODEL_ABBR = {
    "random_forest": "rf",
    "xgboost":       "xgb",
    "mlp":           "mlp",
    "lstm":          "lstm",
    "cnn":           "cnn",
}

MODEL_REGISTRY = {
    "random_forest": "classical",
    "xgboost":       "classical",
    "mlp":           "pytorch",
    "lstm":          "pytorch",
    "cnn":           "pytorch",   # Step 1 only (2184-feature input)
}

_STEP_NAMES = {
    1: "MESA Full PSG",
    2: "MESA Aligned Features",
    3: "Cross-Dataset: MESA → TIHM (raw)",
    4: "Cross-Dataset: MESA → TIHM (harmonized)",
}


# =============================================================================
# Internal helpers
# =============================================================================

def _p(key: str) -> Path:
    return ROOT_DIR / CONFIG["paths"][key]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE]


def _fmt_dist(y: np.ndarray) -> str:
    n = len(y)
    parts = [
        f"{name} {int((y == idx).sum()) / n * 100:.1f}%"
        for idx, name in enumerate(CONFIG["label_names"])
    ]
    return " / ".join(parts)


def _encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].map(CONFIG["label_map"])
    return df


def _clean_nans(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  [info] Dropped {dropped:,} rows with NaN features", flush=True)
    return df


# =============================================================================
# Data loaders
# =============================================================================

def load_step1() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all MESA full PSG feature CSVs (outputs/features/full/).
    Returns X (float32 ndarray), y (int ndarray), groups (subject_id ndarray).
    NaN and Inf values are imputed so PyTorch models receive clean input.
    """
    full_dir = _p("mesa_full")
    files    = sorted(full_dir.glob("mesa_features_full_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {full_dir}")

    frames = []
    for f in files:
        sid = f.stem.rsplit("_", 1)[-1]   # mesa_features_full_0001 → 0001
        df  = pd.read_csv(f)
        df["_subject_id"] = sid
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = _encode_labels(combined)
    combined = combined.dropna(subset=["label"]).reset_index(drop=True)

    # Get feature columns only
    meta_cols    = ["label", "_subject_id"]
    feature_cols = [c for c in combined.columns if c not in meta_cols]

    # Report NaN situation
    nan_count = combined[feature_cols].isna().sum().sum()
    nan_cols  = int(combined[feature_cols].isna().any().sum())
    print(f"  [info] TSFEL NaN values: {nan_count:,} across {nan_cols} feature columns",
          flush=True)

    # Fill NaN with column median; entire-NaN columns → 0
    for col in feature_cols:
        col_median = combined[col].median()
        if np.isnan(col_median):
            combined[col] = combined[col].fillna(0.0)
        else:
            combined[col] = combined[col].fillna(col_median)

    # Replace any inf values
    combined[feature_cols] = combined[feature_cols].replace(
        [np.inf, -np.inf], 0.0)

    # Final verification
    nan_after = int(combined[feature_cols].isna().sum().sum())
    inf_after = int(np.isinf(combined[feature_cols].values).sum())
    print(f"  [info] After imputation — NaN: {nan_after}, Inf: {inf_after}",
          flush=True)

    X      = combined[feature_cols].values.astype(np.float32)
    y      = combined["label"].values
    groups = combined["_subject_id"].values

    print(f"  Loaded {len(X):,} epochs from {len(np.unique(groups))} subjects",
          flush=True)
    print(f"  Classes: {_fmt_dist(y)}", flush=True)
    return X, y, groups


def load_step2() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load all MESA aligned feature CSVs (outputs/features/aligned/).
    Returns X (DataFrame), y (int ndarray), groups (subject_id ndarray).
    """
    aligned_dir = _p("mesa_aligned")
    files       = sorted(aligned_dir.glob("mesa_aligned_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {aligned_dir}")

    frames = []
    for f in files:
        sid = f.stem.rsplit("_", 1)[-1]   # mesa_aligned_0001 → 0001
        df  = pd.read_csv(f)
        df["_subject_id"] = sid
        frames.append(df)

    df        = pd.concat(frames, ignore_index=True)
    df        = _encode_labels(df)
    feat_cols = [c for c in _feature_cols(df) if c != "_subject_id"]
    df        = _clean_nans(df, feat_cols)

    X      = df[feat_cols]
    y      = df["label"].values.astype(int)
    groups = df["_subject_id"].values

    print(f"  Loaded {len(X):,} epochs from {len(np.unique(groups))} subjects",
          flush=True)
    print(f"  Classes: {_fmt_dist(y)}", flush=True)
    return X, y, groups


def load_step3() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Train: MESA aligned CSVs.  Test: TIHM aligned combined CSV.
    Returns X_train, y_train, X_test, y_test.
    """
    aligned_dir = _p("mesa_aligned")
    files       = sorted(aligned_dir.glob("mesa_aligned_*.csv"))
    if not files:
        raise FileNotFoundError(f"No MESA aligned CSVs in {aligned_dir}")

    train_df  = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    train_df  = _encode_labels(train_df)
    feat_cols = _feature_cols(train_df)
    train_df  = _clean_nans(train_df, feat_cols)

    tihm_path = _p("tihm_aligned")
    if not tihm_path.exists():
        raise FileNotFoundError(f"TIHM aligned CSV not found: {tihm_path}")
    test_df = pd.read_csv(tihm_path)
    test_df = _encode_labels(test_df)
    test_df = _clean_nans(test_df, feat_cols)

    X_train = train_df[feat_cols]
    y_train = train_df["label"].values.astype(int)
    X_test  = test_df[feat_cols]
    y_test  = test_df["label"].values.astype(int)

    print(f"  Train (MESA aligned) : {len(X_train):,} epochs  {_fmt_dist(y_train)}",
          flush=True)
    print(f"  Test  (TIHM aligned) : {len(X_test):,} epochs  {_fmt_dist(y_test)}",
          flush=True)
    return X_train, y_train, X_test, y_test


def load_step4() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Train: MESA harmonized.  Test: TIHM harmonized.
    Requires script 08 to have generated outputs/features/harmonized/.
    """
    harm_dir  = ROOT_DIR / "outputs/features/harmonized"
    mesa_path = harm_dir / "mesa_harmonized_all.csv"
    tihm_path = harm_dir / "tihm_harmonized_all.csv"

    for p in (mesa_path, tihm_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run script 08 first to generate harmonized features."
            )

    train_df  = pd.read_csv(mesa_path)
    test_df   = pd.read_csv(tihm_path)
    train_df  = _encode_labels(train_df)
    test_df   = _encode_labels(test_df)
    feat_cols = _feature_cols(train_df)
    train_df  = _clean_nans(train_df, feat_cols)
    test_df   = _clean_nans(test_df, feat_cols)

    X_train = train_df[feat_cols]
    y_train = train_df["label"].values.astype(int)
    X_test  = test_df[feat_cols]
    y_test  = test_df["label"].values.astype(int)

    print(f"  Train (MESA harmonized) : {len(X_train):,} epochs  {_fmt_dist(y_train)}",
          flush=True)
    print(f"  Test  (TIHM harmonized) : {len(X_test):,} epochs  {_fmt_dist(y_test)}",
          flush=True)
    return X_train, y_train, X_test, y_test


# =============================================================================
# Preprocessor
# =============================================================================

def feature_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Step 1 only: select top N features by importance from a quick RandomForest.
    Fitted on X_train only — applied to both X_train and X_test.
    Returns (X_train_sel, X_test_sel, selected_feature_names).
    """
    cfg    = CONFIG["feature_selection"]
    n_keep = cfg["n_features"]

    t0 = time.time()
    quick_rf = RandomForestClassifier(
        n_estimators=cfg["quick_estimators"],
        max_features="sqrt",
        n_jobs=-1,
        random_state=CONFIG["seed"],
        class_weight="balanced",
    )
    quick_rf.fit(X_train, y_train)

    importances = quick_rf.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:n_keep]
    top_cols    = [X_train.columns[i] for i in top_idx]
    elapsed     = time.time() - t0
    top5        = ", ".join(top_cols[:5])

    print(f"  Feature selection: {n_keep} / {len(X_train.columns)} features selected"
          f"  ({elapsed:.0f}s)", flush=True)
    print(f"  Top 5: {top5}", flush=True)

    return X_train[top_cols], X_test[top_cols], top_cols


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Balanced class weights computed from training fold only."""
    classes = np.unique(y_train)
    weights = _sklearn_cw("balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


# =============================================================================
# Model registry
# =============================================================================

def get_model(model_name: str, input_dim: int = 17, class_weights=None):
    if model_name == "random_forest":
        params = CONFIG["random_forest"].copy()
        return RandomForestClassifier(**params)

    elif model_name == "xgboost":
        if not _XGBOOST_OK:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        params = CONFIG["xgboost"].copy()
        if class_weights:
            # XGBoost uses sample_weight not class_weight
            # weights applied during fit, not here
            pass
        return XGBClassifier(**params)

    elif model_name == "mlp":
        if not _TORCH_OK:
            raise ImportError("torch not installed. See requirements.txt.")
        cfg = CONFIG["mlp"]
        return SleepMLP(
            input_dim=input_dim,
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        )

    elif model_name == "lstm":
        if not _TORCH_OK:
            raise ImportError("torch not installed. See requirements.txt.")
        cfg = CONFIG["lstm"]
        return SleepLSTM(
            input_dim=input_dim,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"],
        )

    elif model_name == "cnn":
        if not _TORCH_OK:
            raise ImportError("torch not installed. See requirements.txt.")
        cfg = CONFIG["cnn"]
        return SleepCNN(input_dim=input_dim, dropout=cfg["dropout"])

    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy, per-class F1, macro F1, ROC-AUC, PR-AUC."""
    # Clip to avoid log(0), then re-normalise so rows still sum to exactly 1.0.
    # sklearn roc_auc_score does a strict sum==1 check; clipping alone breaks it.
    y_proba = np.clip(y_proba, 1e-7, 1 - 1e-7)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_cls  = f1_score(y_true, y_pred, average=None, zero_division=0,
                        labels=[0, 1, 2, 3])

    try:
        present   = np.unique(y_true)
        n_classes = y_proba.shape[1]
        # Cast to float64: float32 division leaves sums ~1e-7 off 1.0, which
        # fails sklearn's strict atol=1e-8 check inside roc_auc_score.
        if len(present) == n_classes:
            p64 = y_proba.astype(np.float64)
            p64 = p64 / p64.sum(axis=1, keepdims=True)
            roc_auc = float(roc_auc_score(
                y_true, p64, multi_class="ovr", average="macro"
            ))
        else:
            # Missing classes — slice to present classes and renorm that subset
            p64 = y_proba[:, present].astype(np.float64)
            p64 = p64 / p64.sum(axis=1, keepdims=True)
            roc_auc = float(roc_auc_score(
                y_true, p64, multi_class="ovr", average="macro",
                labels=present,
            ))
    except Exception as e:
        roc_auc = float("nan")
        print(f"  [warn] ROC-AUC failed: {e}", flush=True)

    try:
        pr_auc = float(np.mean([
            average_precision_score((y_true == i).astype(int), y_proba[:, i])
            for i in range(len(CONFIG["label_names"]))
        ]))
    except Exception:
        pr_auc = float("nan")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_awake": float(per_cls[0]),
        "f1_light": float(per_cls[1]),
        "f1_deep":  float(per_cls[2]),
        "f1_rem":   float(per_cls[3]),
        "roc_auc":  roc_auc,
        "pr_auc":   pr_auc,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step_num: int,
    model_name: str,
) -> Path:
    label_names = CONFIG["label_names"]
    abbr        = _MODEL_ABBR[model_name]
    out_path    = _p("figures") / f"confusion_matrix_step{step_num}_{abbr}.svg"

    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Step {step_num} — {model_name.replace('_', ' ').title()}")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def plot_cv_boxplot(
    metrics_df: pd.DataFrame,
    step_num: int,
    model_name: str,
) -> Path:
    abbr     = _MODEL_ABBR[model_name]
    out_path = _p("figures") / f"cv_metrics_step{step_num}_{abbr}.svg"
    cols     = ["accuracy", "macro_f1", "f1_awake", "f1_light",
                "f1_deep", "f1_rem", "roc_auc", "pr_auc"]
    present  = [c for c in cols if c in metrics_df.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([metrics_df[c].values for c in present], tick_labels=present,
               patch_artist=True)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Step {step_num} CV metrics — {model_name.replace('_', ' ').title()}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def plot_feature_importance(
    model,
    feature_names: list[str],
    step_num: int,
    model_name: str,
    top_n: int = 30,
) -> Path | None:
    if not hasattr(model, "feature_importances_"):
        return None

    top_n    = min(top_n, len(feature_names))
    abbr     = _MODEL_ABBR[model_name]
    out_path = _p("figures") / f"feature_importance_step{step_num}_{abbr}.svg"

    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:top_n]
    top_names   = [feature_names[i] for i in top_idx]
    top_vals    = importances[top_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3 + 1)))
    ax.barh(range(top_n), top_vals[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Step {step_num} — Top {top_n} features"
        f" — {model_name.replace('_', ' ').title()}"
    )
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    step_num: int,
    model_name: str,
) -> Path:
    abbr        = _MODEL_ABBR[model_name]
    out_path    = _p("figures") / f"roc_curves_step{step_num}_{abbr}.svg"
    label_names = CONFIG["label_names"]
    colors      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls_idx, (name, color) in enumerate(zip(label_names, colors)):
        y_bin = (y_true == cls_idx).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, cls_idx])
            auc_val     = roc_auc_score(y_bin, y_proba[:, cls_idx])
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")
        except ValueError:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Step {step_num} ROC — {model_name.replace('_', ' ').title()}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


# =============================================================================
# Logging setup
# =============================================================================

def _setup_logger(step_num: int, model_name: str) -> logging.Logger:
    abbr     = _MODEL_ABBR.get(model_name, model_name)
    log_dir  = _p("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"step{step_num}_{abbr}.log"

    logger = logging.getLogger(f"step{step_num}_{abbr}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return logger


# =============================================================================
# TensorBoard
# =============================================================================

def _make_writer(step_num: int, model_name: str):
    if not _TB_OK:
        return None
    abbr   = _MODEL_ABBR.get(model_name, model_name)
    tb_dir = _p("tensorboard") / f"step{step_num}_{abbr}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(tb_dir))


def _tb_log_metrics(writer, metrics: dict, fold: int) -> None:
    if writer is None:
        return
    for key, val in metrics.items():
        if not (isinstance(val, float) and np.isnan(val)):
            writer.add_scalar(f"fold/{key}", val, global_step=fold)


def _tb_log_feature_importance(
    writer,
    model,
    feature_names: list[str],
    step_num: int,
    model_name: str,
) -> None:
    if writer is None or not hasattr(model, "feature_importances_"):
        return

    # SVG saved to outputs/figures/ for thesis; figure is closed inside plot_feature_importance
    fig_path = plot_feature_importance(model, feature_names, step_num, model_name)
    if fig_path is None or not fig_path.exists():
        return

    # Render a separate PNG for TensorBoard (plt.imread only works with raster formats)
    png_path = fig_path.with_suffix(".png")
    top_n    = min(30, len(feature_names))
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:top_n]
    top_names   = [feature_names[i] for i in top_idx]
    top_vals    = importances[top_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3 + 1)))
    ax.barh(range(top_n), top_vals[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Step {step_num} — Top {top_n} features"
        f" — {model_name.replace('_', ' ').title()}"
    )
    plt.tight_layout()
    fig.savefig(png_path, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    img = plt.imread(str(png_path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    writer.add_image(
        f"feature_importance/step{step_num}_{model_name}",
        img.transpose(2, 0, 1),
        global_step=0,
    )


# =============================================================================
# Training — CV steps (1 and 2)
# =============================================================================

def run_cv_step(
    step_num: int,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> None:
    abbr    = _MODEL_ABBR[model_name]
    logger  = _setup_logger(step_num, model_name)
    writer  = _make_writer(step_num, model_name)
    n_folds = CONFIG["n_folds"]
    gkf     = GroupKFold(n_splits=n_folds)

    # load_step1 returns numpy; load_step2 returns DataFrame — handle both.
    X_is_df = isinstance(X, pd.DataFrame)
    if X_is_df:
        last_feat_cols = list(X.columns)
        do_feat_sel    = step_num == 1 and CONFIG["feature_selection"]["enabled"]
    else:
        last_feat_cols = list(range(X.shape[1]))
        do_feat_sel    = False   # feature_selection requires DataFrame column names

    logger.info(f"Step {step_num} | {model_name} | X={X.shape} | "
                f"subjects={len(np.unique(groups))}")

    fold_metrics  = []
    all_y_true    = []
    all_y_pred    = []
    pred_parts    = []
    last_model    = None

    t0_step = time.time()

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups), start=1
    ):
        t0_fold = time.time()

        if X_is_df:
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_te, y_te = X.iloc[test_idx],  y[test_idx]
        else:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx],  y[test_idx]

        # Safety net — should be zero after load_step1 imputation
        if np.isnan(X_tr).sum() > 0 or np.isnan(X_te).sum() > 0:
            print(f"  [warn] fold {fold} has NaN after load — applying nan_to_num",
                  flush=True)
            X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
            X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        if do_feat_sel:
            if fold == 1:
                X_tr, X_te, selected_cols = feature_selection(X_tr, y_tr, X_te)
            else:
                X_tr = X_tr[last_feat_cols]
                X_te = X_te[last_feat_cols]
                selected_cols = last_feat_cols
            last_feat_cols = selected_cols

        cw    = compute_class_weights(y_tr)
        model = get_model(model_name)

        # Ensure arrays are numpy for both sklearn and XGBoost
        X_tr_arr = X_tr.values if hasattr(X_tr, "values") else X_tr
        X_te_arr = X_te.values if hasattr(X_te, "values") else X_te

        # Normalise for PyTorch models — TSFEL features span many orders of
        # magnitude which causes overflow in the forward pass without scaling.
        # Tree-based models are scale-invariant so no scaling is applied there.
        if model_name in ("mlp", "lstm", "cnn"):
            from sklearn.preprocessing import StandardScaler
            scaler   = StandardScaler()
            X_tr_arr = scaler.fit_transform(X_tr_arr)
            X_te_arr = scaler.transform(X_te_arr)
            # Zero-variance columns produce NaN after scaling — replace with 0
            X_tr_arr = np.nan_to_num(X_tr_arr, nan=0.0, posinf=0.0, neginf=0.0)
            X_te_arr = np.nan_to_num(X_te_arr, nan=0.0, posinf=0.0, neginf=0.0)

        if model_name == "xgboost":
            sample_weight = np.array([cw[c] for c in y_tr])
            model.fit(X_tr_arr, y_tr, sample_weight=sample_weight)
        else:
            model.fit(X_tr_arr, y_tr)

        y_pred  = model.predict(X_te_arr)
        y_proba = model.predict_proba(X_te_arr)

        m = compute_metrics(y_te, y_pred, y_proba)
        fold_metrics.append(m)

        all_y_true.append(y_te)
        all_y_pred.append(y_pred)

        pred_df = pd.DataFrame(
            y_proba,
            columns=[f"proba_{n}" for n in CONFIG["label_names"]],
        )
        pred_df["y_true"] = y_te
        pred_df["y_pred"] = y_pred
        pred_df["fold"]   = fold
        pred_parts.append(pred_df)

        _tb_log_metrics(writer, m, fold)
        last_model = model

        elapsed_fold = (time.time() - t0_fold) / 60
        print(f"  Fold {fold}/{n_folds} ... done"
              f"  (acc {m['accuracy']:.3f}  f1 {m['macro_f1']:.3f})"
              f"  {elapsed_fold:.1f} min", flush=True)
        logger.info(f"Fold {fold} | acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}"
                    f" roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f}")

    total_min = (time.time() - t0_step) / 60

    # Aggregate
    metrics_df = pd.DataFrame(fold_metrics)
    means      = metrics_df.mean()
    stds       = metrics_df.std()

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    # Save predictions
    _p("predictions").mkdir(parents=True, exist_ok=True)
    pred_path = _p("predictions") / f"step{step_num}_{abbr}_predictions.csv"
    pd.concat(pred_parts, ignore_index=True).to_csv(pred_path, index=False)

    # Save metrics
    _p("results").mkdir(parents=True, exist_ok=True)
    metrics_path = _p("results") / f"step{step_num}_{abbr}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Plots
    _p("figures").mkdir(parents=True, exist_ok=True)
    cm_path  = plot_confusion_matrix(all_y_true, all_y_pred, step_num, model_name)
    box_path = plot_cv_boxplot(metrics_df, step_num, model_name)
    fi_path  = plot_feature_importance(last_model, last_feat_cols, step_num, model_name)
    _tb_log_feature_importance(writer, last_model, last_feat_cols, step_num, model_name)

    # Save model
    _p("models").mkdir(parents=True, exist_ok=True)
    model_path = _p("models") / f"step{step_num}_{abbr}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(last_model, fh)

    if writer:
        writer.close()

    # Summary
    print(f"\n  Results:", flush=True)
    for key, label in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("f1_awake", "F1 AWAKE"),
        ("f1_light", "F1 LIGHT"),
        ("f1_deep",  "F1 DEEP "),
        ("f1_rem",   "F1 REM  "),
        ("roc_auc",  "ROC-AUC "),
        ("pr_auc",   "PR-AUC  "),
    ]:
        print(f"   {label} : {means[key]:.3f} ± {stds[key]:.3f}", flush=True)
    print(f"   Time     : {total_min:.1f} min", flush=True)

    print(f"\n  Saved: {metrics_path.relative_to(ROOT_DIR)}", flush=True)
    print(f"  Saved: {cm_path.relative_to(ROOT_DIR)}", flush=True)
    print(f"  Saved: {box_path.relative_to(ROOT_DIR)}", flush=True)
    if fi_path:
        print(f"  Saved: {fi_path.relative_to(ROOT_DIR)}", flush=True)
    print(f"  Saved: {model_path.relative_to(ROOT_DIR)}", flush=True)

    for key, val in [
        ("accuracy", means["accuracy"]), ("macro_f1", means["macro_f1"]),
        ("roc_auc",  means["roc_auc"]),
    ]:
        logger.info(f"Final | {key}={val:.4f}±{stds[key]:.4f}")


# =============================================================================
# Training — cross-dataset steps (3 and 4)
# =============================================================================

def run_cross_dataset_step(
    step_num: int,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> None:
    abbr   = _MODEL_ABBR[model_name]
    logger = _setup_logger(step_num, model_name)
    writer = _make_writer(step_num, model_name)

    logger.info(f"Step {step_num} | {model_name} | "
                f"train={X_train.shape} test={X_test.shape}")

    cw    = compute_class_weights(y_train)
    model = get_model(model_name)

    t0 = time.time()
    if model_name == "xgboost":
        sample_weight = np.array([cw[c] for c in y_train])
        model.fit(X_train.values, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    elapsed = (time.time() - t0) / 60

    m = compute_metrics(y_test, y_pred, y_proba)

    # Save predictions
    _p("predictions").mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        y_proba,
        columns=[f"proba_{n}" for n in CONFIG["label_names"]],
    )
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    pred_path = _p("predictions") / f"step{step_num}_{abbr}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save metrics
    _p("results").mkdir(parents=True, exist_ok=True)
    metrics_path = _p("results") / f"step{step_num}_{abbr}_metrics.csv"
    pd.DataFrame([m]).to_csv(metrics_path, index=False)

    _tb_log_metrics(writer, m, fold=0)

    # Plots
    _p("figures").mkdir(parents=True, exist_ok=True)
    cm_path  = plot_confusion_matrix(y_test, y_pred, step_num, model_name)
    roc_path = plot_roc_curves(y_test, y_proba, step_num, model_name)
    feat_names = list(X_train.columns)
    fi_path    = plot_feature_importance(model, feat_names, step_num, model_name)
    _tb_log_feature_importance(writer, model, feat_names, step_num, model_name)

    # Save model
    _p("models").mkdir(parents=True, exist_ok=True)
    model_path = _p("models") / f"step{step_num}_{abbr}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    if writer:
        writer.close()

    # Summary
    print(f"\n  Results:", flush=True)
    for key, label in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("f1_awake", "F1 AWAKE"),
        ("f1_light", "F1 LIGHT"),
        ("f1_deep",  "F1 DEEP "),
        ("f1_rem",   "F1 REM  "),
        ("roc_auc",  "ROC-AUC "),
        ("pr_auc",   "PR-AUC  "),
    ]:
        print(f"   {label} : {m[key]:.3f}", flush=True)
    print(f"   Time     : {elapsed:.1f} min", flush=True)

    print(f"\n  Saved: {metrics_path.relative_to(ROOT_DIR)}", flush=True)
    print(f"  Saved: {cm_path.relative_to(ROOT_DIR)}", flush=True)
    print(f"  Saved: {roc_path.relative_to(ROOT_DIR)}", flush=True)
    if fi_path:
        print(f"  Saved: {fi_path.relative_to(ROOT_DIR)}", flush=True)
    print(f"  Saved: {model_path.relative_to(ROOT_DIR)}", flush=True)

    logger.info(f"Final | acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}"
                f" roc_auc={m['roc_auc']:.4f}")


# =============================================================================
# Training — PyTorch steps (MLP, LSTM, CNN)
# =============================================================================

def run_pytorch_step(
    step_num: int,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
) -> None:
    """
    PyTorch training for MLP, LSTM, CNN.
    - If groups is provided: GroupKFold CV (steps 1, 2).
    - If X_test/y_test provided: single train→test split (steps 3, 4).
    """
    if not _TORCH_OK:
        raise ImportError("torch not installed. See requirements.txt.")

    abbr       = _MODEL_ABBR[model_name]
    logger     = _setup_logger(step_num, model_name)
    writer     = _make_writer(step_num, model_name)
    cfg        = CONFIG[model_name]
    device     = _DEVICE
    n_folds    = CONFIG["n_folds"]
    input_dim  = X.shape[1]
    is_cv      = groups is not None and X_test is None

    logger.info(f"Step {step_num} | {model_name} | X={X.shape} | device={device}")
    print(f"  Device: {device}", flush=True)

    fold_metrics  = []
    all_y_true    = []
    all_y_pred    = []
    pred_parts    = []
    last_model    = None
    t0_step       = time.time()

    def _run_fold(fold, X_tr, y_tr, X_te, y_te):
        nonlocal last_model

        # Normalise features — fit on training fold only, apply to both
        from sklearn.preprocessing import StandardScaler
        _scaler  = StandardScaler()
        X_tr_arr = _scaler.fit_transform(
            X_tr.values if hasattr(X_tr, "values") else X_tr
        )
        X_te_arr = _scaler.transform(
            X_te.values if hasattr(X_te, "values") else X_te
        )
        # Zero-variance columns produce NaN after scaling — replace with 0
        X_tr_arr = np.nan_to_num(X_tr_arr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_arr = np.nan_to_num(X_te_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Class weights → criterion
        cw_dict  = compute_class_weights(y_tr)
        cw_tensor = torch.FloatTensor(
            [cw_dict.get(i, 1.0) for i in range(len(CONFIG["label_names"]))]
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)

        # Datasets and loaders — use scaled arrays
        if model_name == "lstm":
            seq_len  = cfg["seq_len"]
            train_ds = SequenceDataset(X_tr_arr, y_tr, seq_len=seq_len)
            val_ds   = SequenceDataset(X_te_arr, y_te, seq_len=seq_len)
        else:
            train_ds = EpochDataset(X_tr_arr, y_tr)
            val_ds   = EpochDataset(X_te_arr, y_te)

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                                  shuffle=False, num_workers=0, pin_memory=True)

        model = get_model(model_name, input_dim=input_dim).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        t0_fold = time.time()
        model, history = train_pytorch_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, criterion=criterion, scheduler=scheduler,
            device=device, epochs=cfg["epochs"], patience=cfg["patience"],
            writer=writer, fold=fold,
        )

        # Predict on validation set
        model.eval()
        all_preds, all_proba = [], []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(X_batch)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_proba.append(proba)

        y_pred  = np.concatenate(all_preds)
        y_proba = np.concatenate(all_proba)

        # Align y_te to valid indices (SequenceDataset skips first seq_len-1 rows)
        if model_name == "lstm":
            y_te_aligned = y_te[val_ds.valid_idx]
        else:
            y_te_aligned = y_te

        m = compute_metrics(y_te_aligned, y_pred, y_proba)
        elapsed_fold = (time.time() - t0_fold) / 60

        all_y_true.append(y_te_aligned)
        all_y_pred.append(y_pred)

        pred_df = pd.DataFrame(y_proba,
                               columns=[f"proba_{n}" for n in CONFIG["label_names"]])
        pred_df["y_true"] = y_te_aligned
        pred_df["y_pred"] = y_pred
        pred_df["fold"]   = fold
        pred_parts.append(pred_df)

        _tb_log_metrics(writer, m, fold)
        last_model = model

        # Save per-fold checkpoint
        _p("models").mkdir(parents=True, exist_ok=True)
        ckpt_path = _p("models") / f"step{step_num}_{abbr}_fold{fold}.pt"
        torch.save(model.state_dict(), ckpt_path)

        label_str = f"fold {fold}/{n_folds}" if is_cv else "train→test"
        print(f"  {label_str} ... done"
              f"  (acc {m['accuracy']:.3f}  f1 {m['macro_f1']:.3f})"
              f"  {elapsed_fold:.1f} min", flush=True)
        logger.info(f"Fold {fold} | acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}"
                    f" roc_auc={m['roc_auc']:.4f}")
        return m

    # ── CV or single split ────────────────────────────────────────────────────
    if is_cv:
        gkf = GroupKFold(n_splits=n_folds)
        for fold, (tr_idx, te_idx) in enumerate(
            gkf.split(X, y, groups), start=1
        ):
            X_tr = X.iloc[tr_idx] if isinstance(X, pd.DataFrame) else X[tr_idx]
            X_te = X.iloc[te_idx] if isinstance(X, pd.DataFrame) else X[te_idx]
            m = _run_fold(fold, X_tr, y[tr_idx], X_te, y[te_idx])
            fold_metrics.append(m)
    else:
        m = _run_fold(1, X, y, X_test, y_test)
        fold_metrics.append(m)

    # ── Aggregate and save ────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(fold_metrics)
    means      = metrics_df.mean()
    stds       = metrics_df.std().fillna(0)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    _p("predictions").mkdir(parents=True, exist_ok=True)
    pred_path = _p("predictions") / f"step{step_num}_{abbr}_predictions.csv"
    pd.concat(pred_parts, ignore_index=True).to_csv(pred_path, index=False)

    _p("results").mkdir(parents=True, exist_ok=True)
    metrics_path = _p("results") / f"step{step_num}_{abbr}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    _p("figures").mkdir(parents=True, exist_ok=True)
    cm_path = plot_confusion_matrix(all_y_true, all_y_pred, step_num, model_name)
    if is_cv:
        plot_cv_boxplot(metrics_df, step_num, model_name)
    else:
        plot_roc_curves(all_y_true,
                        pd.concat(pred_parts)[[f"proba_{n}"
                                               for n in CONFIG["label_names"]]].values,
                        step_num, model_name)

    if writer:
        writer.close()

    total_min = (time.time() - t0_step) / 60
    print(f"\n  Results:", flush=True)
    sep_label = "± " if is_cv else "  "
    for key, label in [
        ("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
        ("f1_awake", "F1 AWAKE"), ("f1_light", "F1 LIGHT"),
        ("f1_deep",  "F1 DEEP "), ("f1_rem",   "F1 REM  "),
        ("roc_auc",  "ROC-AUC "), ("pr_auc",   "PR-AUC  "),
    ]:
        if is_cv:
            print(f"   {label} : {means[key]:.3f} ± {stds[key]:.3f}", flush=True)
        else:
            print(f"   {label} : {means[key]:.3f}", flush=True)
    print(f"   Time     : {total_min:.1f} min", flush=True)
    print(f"\n  Saved: {metrics_path.relative_to(ROOT_DIR)}", flush=True)
    print(f"  Saved: {cm_path.relative_to(ROOT_DIR)}", flush=True)

    logger.info(f"Final | acc={means['accuracy']:.4f} macro_f1={means['macro_f1']:.4f}")


# =============================================================================
# Top-level dispatch
# =============================================================================

def run_step(step_num: int, model_name: str, resume: bool = False) -> None:
    abbr         = _MODEL_ABBR.get(model_name, model_name)
    results_path = _p("results") / f"step{step_num}_{abbr}_metrics.csv"

    if resume and results_path.exists():
        print(f"  [skip] Step {step_num} {model_name} — results already exist",
              flush=True)
        return

    SEP = "\u2550" * 42
    print(f"\n{SEP}", flush=True)
    print(f" STEP {step_num} — {_STEP_NAMES[step_num]}", flush=True)
    print(f" Model: {model_name.replace('_', ' ').title()}", flush=True)
    print(SEP, flush=True)

    is_pytorch = MODEL_REGISTRY.get(model_name) == "pytorch"

    if step_num == 1:
        X, y, groups = load_step1()
        if is_pytorch:
            run_pytorch_step(step_num, model_name, X, y, groups)
        else:
            run_cv_step(step_num, model_name, X, y, groups)

    elif step_num == 2:
        X, y, groups = load_step2()
        if is_pytorch:
            run_pytorch_step(step_num, model_name, X, y, groups)
        else:
            run_cv_step(step_num, model_name, X, y, groups)

    elif step_num == 3:
        X_train, y_train, X_test, y_test = load_step3()
        if is_pytorch:
            run_pytorch_step(step_num, model_name, X_train, y_train,
                             groups=None, X_test=X_test, y_test=y_test)
        else:
            run_cross_dataset_step(step_num, model_name,
                                   X_train, y_train, X_test, y_test)

    elif step_num == 4:
        X_train, y_train, X_test, y_test = load_step4()
        if is_pytorch:
            run_pytorch_step(step_num, model_name, X_train, y_train,
                             groups=None, X_test=X_test, y_test=y_test)
        else:
            run_cross_dataset_step(step_num, model_name,
                                   X_train, y_train, X_test, y_test)

    print(SEP, flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="python scripts/07_train_models.py",
        description="Sleep-stage ML training pipeline.",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                     help="Run a specific step (1–4).")
    grp.add_argument("--all", action="store_true",
                     help="Run all steps sequentially.")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["random_forest", "xgboost", "mlp", "lstm", "cnn", "all"],
        help="Model to train (default: all → rf + xgb + mlp + lstm + cnn).",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps where results CSV already exists.")
    args = parser.parse_args()

    models_to_run = (
        ["random_forest", "xgboost", "mlp", "lstm", "cnn"]
        if args.model == "all" else [args.model]
    )
    steps_to_run  = [1, 2, 3, 4] if args.all else [args.step]

    for step in steps_to_run:
        for model_name in models_to_run:
            run_step(step, model_name, args.resume)


if __name__ == "__main__":
    main()
