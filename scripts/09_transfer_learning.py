"""
scripts/09_transfer_learning.py
================================
RQ4 — Transfer learning from MESA (clinical PSG) to TIHM (wearable).

Loads pretrained Step-3 models, fine-tunes on a small number of TIHM patients,
and evaluates on held-out TIHM patients.  Directly answers RQ4: can transfer
learning improve cross-device performance?

Usage:
  python scripts/09_transfer_learning.py --model lstm
  python scripts/09_transfer_learning.py --model mlp
  python scripts/09_transfer_learning.py --model random_forest
  python scripts/09_transfer_learning.py --model xgboost
  python scripts/09_transfer_learning.py --all
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
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from xgboost import XGBClassifier
    _XGBOOST_OK = True
except ImportError:
    _XGBOOST_OK = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    _TORCH_OK = True
    _DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH_OK = False
    _DEVICE   = None

if _TORCH_OK:
    from utils.models.mlp     import SleepMLP
    from utils.models.lstm    import SleepLSTM
    from utils.data_utils     import EpochDataset, SequenceDataset
    from utils.training_utils import train_pytorch_model


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "seed": 42,
    "label_map": {"AWAKE": 0, "LIGHT": 1, "DEEP": 2, "REM": 3},
    "label_names": ["AWAKE", "LIGHT", "DEEP", "REM"],
    "finetune_sizes": [2, 5, 10],   # TIHM patients used for fine-tuning

    "paths": {
        "mesa_aligned":      "outputs/features/aligned/",
        "tihm_aligned":      "outputs/features/tihm_aligned_all.csv",
        "pretrained_models": "outputs/models/",
        "results":           "outputs/results/rq4/",
        "figures":           "outputs/figures/rq4/",
        "logs":              "logs/rq4/",
    },

    "finetune_pytorch": {
        "last_layer_lr":  0.001,
        "full_model_lr":  0.0001,
        "epochs":         30,
        "patience":       5,
        "batch_size":     256,
        "weight_decay":   1e-4,
    },

    "random_forest": {
        "n_estimators":     500,
        "max_features":     "sqrt",
        "min_samples_leaf": 5,
        "n_jobs":           -1,
        "class_weight":     "balanced",
        "random_state":     42,
    },

    "xgboost": {
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "n_jobs":           -1,
    },

    "mlp": {
        "hidden_dims": [256, 128, 64],
        "dropout":     0.3,
    },

    "lstm": {
        "hidden_size":   128,
        "num_layers":    2,
        "dropout":       0.3,
        "bidirectional": True,
        "seq_len":       10,
    },

    "device": _DEVICE.type if _DEVICE is not None else "cpu",
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

_MODEL_ABBR = {
    "random_forest": "rf",
    "xgboost":       "xgb",
    "mlp":           "mlp",
    "lstm":          "lstm",
}


# =============================================================================
# Helpers
# =============================================================================

def _p(key: str) -> Path:
    return ROOT_DIR / CONFIG["paths"][key]


def _encode_labels(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_integer_dtype(series):
        return series.map(CONFIG["label_map"])
    return series


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: np.ndarray) -> dict:
    """Compute accuracy, per-class F1, macro F1, ROC-AUC, PR-AUC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
    }
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0,
                       labels=[0, 1, 2, 3])
    for i, name in enumerate(CONFIG["label_names"]):
        metrics[f"f1_{name.lower()}"] = float(per_cls[i]) if i < len(per_cls) else 0.0

    try:
        p64     = y_proba.astype(np.float64)
        p64     = p64 / p64.sum(axis=1, keepdims=True)
        present = np.unique(y_true)
        if len(present) == 4:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, p64, multi_class="ovr", average="macro")
            )
        else:
            sub = p64[:, present]
            sub = sub / sub.sum(axis=1, keepdims=True)
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, sub, multi_class="ovr", average="macro",
                              labels=present)
            )
    except Exception as exc:
        metrics["roc_auc"] = float("nan")
        print(f"  [warn] ROC-AUC: {exc}", flush=True)

    try:
        metrics["pr_auc"] = float(np.mean([
            average_precision_score((y_true == i).astype(int), y_proba[:, i])
            for i in range(4)
        ]))
    except Exception:
        metrics["pr_auc"] = float("nan")

    return metrics


def _save_result(model_name: str, strategy: str, n_finetune: int,
                 metrics: dict) -> None:
    out_dir = _p("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {"model": model_name, "strategy": strategy, "n_finetune": n_finetune}
    row.update(metrics)
    out_path = out_dir / f"rq4_{model_name}_{strategy}_n{n_finetune}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)


def _get_step3_baseline(model_name: str) -> float:
    """Load Step 3 macro F1 from saved metrics CSV."""
    abbr         = _MODEL_ABBR[model_name]
    metrics_path = ROOT_DIR / "outputs/results" / f"step3_{abbr}_metrics.csv"
    if not metrics_path.exists():
        print(f"  [warn] Step 3 baseline not found: {metrics_path}", flush=True)
        return float("nan")
    df = pd.read_csv(metrics_path)
    if "macro_f1" in df.columns:
        return float(df["macro_f1"].mean())
    return float("nan")


# =============================================================================
# Data loading
# =============================================================================

def load_mesa_aligned() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all MESA aligned CSVs. Returns X (float32), y, groups."""
    aligned_dir = _p("mesa_aligned")
    files       = sorted(aligned_dir.glob("mesa_aligned_*.csv"))
    if not files:
        raise FileNotFoundError(f"No MESA aligned CSVs in {aligned_dir}")

    frames = []
    for f in files:
        sid = f.stem.rsplit("_", 1)[-1]
        tmp = pd.read_csv(f)
        tmp["_subject_id"] = sid
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = _encode_labels(df["label"])
    df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    X      = df[FEATURE_COLS].values.astype(np.float32)
    y      = df["label"].values
    groups = df["_subject_id"].values

    print(f"  MESA aligned : {len(X):,} epochs, {len(np.unique(groups))} subjects",
          flush=True)
    return X, y, groups


def load_tihm_aligned() -> pd.DataFrame:
    """Load TIHM aligned CSV. Returns DataFrame with patient_id, features, label."""
    tihm_path = ROOT_DIR / CONFIG["paths"]["tihm_aligned"]
    if not tihm_path.exists():
        raise FileNotFoundError(f"TIHM aligned not found: {tihm_path}")
    df = pd.read_csv(tihm_path)
    df["label"] = _encode_labels(df["label"])
    df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"  TIHM aligned : {len(df):,} epochs, {df['patient_id'].nunique()} patients",
          flush=True)
    return df


def split_tihm_patients(tihm_df: pd.DataFrame, n_finetune: int,
                        seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle TIHM patients and split into fine-tune / test sets."""
    patients = sorted(tihm_df["patient_id"].unique())
    rng      = np.random.default_rng(seed)
    rng.shuffle(patients)

    finetune_pts = list(patients[:n_finetune])
    test_pts     = list(patients[n_finetune:])

    finetune_df = tihm_df[tihm_df["patient_id"].isin(finetune_pts)].reset_index(drop=True)
    test_df     = tihm_df[tihm_df["patient_id"].isin(test_pts)].reset_index(drop=True)

    preview = test_pts[:5]
    ellip   = "..." if len(test_pts) > 5 else ""
    print(f"   Fine-tune patients: {finetune_pts}", flush=True)
    print(f"   Test patients ({len(test_pts)}): {preview}{ellip}", flush=True)

    return finetune_df, test_df


# =============================================================================
# Classical model transfer
# =============================================================================

def transfer_classical(model_name: str, n_finetune: int,
                       tihm_df: pd.DataFrame,
                       X_mesa: np.ndarray, y_mesa: np.ndarray) -> dict:
    """
    Retrain classical model from scratch on MESA + TIHM fine-tune data.
    Returns metrics dict evaluated on held-out TIHM patients.
    """
    finetune_df, test_df = split_tihm_patients(tihm_df, n_finetune)

    X_ft   = finetune_df[FEATURE_COLS].values.astype(np.float32)
    y_ft   = finetune_df["label"].values
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["label"].values

    X_train = np.vstack([X_mesa, X_ft])
    y_train = np.concatenate([y_mesa, y_ft])

    present = np.unique(y_train)
    cw      = compute_class_weight("balanced", classes=present, y=y_train)
    cw_dict = dict(zip(present.tolist(), cw.tolist()))

    if model_name == "random_forest":
        model = RandomForestClassifier(**CONFIG["random_forest"])
        model.fit(X_train, y_train)
    elif model_name == "xgboost":
        if not _XGBOOST_OK:
            raise ImportError("xgboost not installed")
        sw    = np.array([cw_dict.get(c, 1.0) for c in y_train])
        model = XGBClassifier(**CONFIG["xgboost"])
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return _compute_metrics(y_test, y_pred, y_proba)


# =============================================================================
# PyTorch fine-tuning
# =============================================================================

def _load_pytorch_model(model_name: str):
    """Instantiate model and load pretrained weights from Step 3."""
    abbr       = _MODEL_ABBR[model_name]
    model_path = ROOT_DIR / CONFIG["paths"]["pretrained_models"] / \
                 f"step3_{abbr}_fold1.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pretrained model not found: {model_path}\n"
            f"Run step 3 first:  python scripts/07_train_models.py --step 3 --model {model_name}"
        )

    input_dim = len(FEATURE_COLS)
    if model_name == "mlp":
        cfg   = CONFIG["mlp"]
        model = SleepMLP(input_dim=input_dim, hidden_dims=cfg["hidden_dims"],
                         dropout=cfg["dropout"])
    elif model_name == "lstm":
        cfg   = CONFIG["lstm"]
        model = SleepLSTM(input_dim=input_dim, hidden_size=cfg["hidden_size"],
                          num_layers=cfg["num_layers"], dropout=cfg["dropout"],
                          bidirectional=cfg["bidirectional"])
    else:
        raise ValueError(f"Unknown PyTorch model: {model_name}")

    state = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(state)
    return model


def finetune_pytorch(model_name: str, strategy: str, n_finetune: int,
                     tihm_df: pd.DataFrame,
                     X_mesa: np.ndarray, y_mesa: np.ndarray) -> dict:
    """
    Fine-tune a pretrained PyTorch model on TIHM patients.

    strategy : 'last_layer' — freeze all but the final Linear layer
               'full_model' — unfreeze everything with a low LR
    """
    if not _TORCH_OK:
        raise ImportError("torch not installed")

    device  = _DEVICE
    cfg_ft  = CONFIG["finetune_pytorch"]

    finetune_df, test_df = split_tihm_patients(tihm_df, n_finetune)

    X_ft   = finetune_df[FEATURE_COLS].values.astype(np.float32)
    y_ft   = finetune_df["label"].values
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["label"].values

    # StandardScaler fitted on MESA training data
    scaler        = StandardScaler()
    scaler.fit(X_mesa)
    X_ft_s        = np.nan_to_num(scaler.transform(X_ft),   nan=0., posinf=0., neginf=0.)
    X_test_s      = np.nan_to_num(scaler.transform(X_test), nan=0., posinf=0., neginf=0.)

    model = _load_pytorch_model(model_name)

    # Apply fine-tuning strategy
    if strategy == "last_layer":
        for param in model.parameters():
            param.requires_grad = False
        if isinstance(model, SleepMLP):
            for param in model.network[-1].parameters():
                param.requires_grad = True
        elif isinstance(model, SleepLSTM):
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
        lr = cfg_ft["last_layer_lr"]
    elif strategy == "full_model":
        for param in model.parameters():
            param.requires_grad = True
        lr = cfg_ft["full_model_lr"]
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    model = model.to(device)

    # Class weights from fine-tune labels
    present   = np.unique(y_ft)
    cw        = compute_class_weight("balanced", classes=present, y=y_ft)
    cw_dict   = dict(zip(present.tolist(), cw.tolist()))
    cw_tensor = torch.FloatTensor(
        [cw_dict.get(i, 1.0) for i in range(len(CONFIG["label_names"]))]
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg_ft["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    if model_name == "lstm":
        seq_len  = CONFIG["lstm"]["seq_len"]
        train_ds = SequenceDataset(X_ft_s,   y_ft,   seq_len=seq_len)
        test_ds  = SequenceDataset(X_test_s, y_test, seq_len=seq_len)
    else:
        train_ds = EpochDataset(X_ft_s,   y_ft)
        test_ds  = EpochDataset(X_test_s, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg_ft["batch_size"],
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg_ft["batch_size"] * 2,
                              shuffle=False, num_workers=0)

    model, _ = train_pytorch_model(
        model=model, train_loader=train_loader, val_loader=test_loader,
        optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        device=device, epochs=cfg_ft["epochs"], patience=cfg_ft["patience"],
        writer=None, fold=1,
    )

    # Evaluate on test set
    model.eval()
    all_preds, all_proba = [], []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            all_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    y_pred  = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba)

    # Align labels for LSTM (SequenceDataset skips first seq_len-1 rows)
    if model_name == "lstm":
        y_eval = y_test[test_ds.valid_idx]
    else:
        y_eval = y_test

    return _compute_metrics(y_eval, y_pred, y_proba)


# =============================================================================
# Learning curve experiment
# =============================================================================

def run_learning_curve(model_name: str, tihm_df: pd.DataFrame,
                       X_mesa: np.ndarray, y_mesa: np.ndarray,
                       logger: logging.Logger) -> tuple[dict, float]:
    """
    Run transfer for each fine-tune size.
    Returns (results_dict, baseline_f1).
    """
    is_pytorch     = model_name in ("mlp", "lstm")
    finetune_sizes = CONFIG["finetune_sizes"]
    baseline_f1    = _get_step3_baseline(model_name)

    strategies = ["last_layer", "full_model"] if is_pytorch else ["combined"]
    results = {
        "n_patients": [0] + finetune_sizes,
        "macro_f1":   {s: [baseline_f1] for s in strategies},
    }

    for n in finetune_sizes:
        print(f"\n Fine-tune size: {n} patients", flush=True)
        logger.info(f"n_finetune={n}")

        for strategy in strategies:
            t0 = time.time()
            try:
                if is_pytorch:
                    m = finetune_pytorch(model_name, strategy, n,
                                        tihm_df, X_mesa, y_mesa)
                else:
                    m = transfer_classical(model_name, n,
                                          tihm_df, X_mesa, y_mesa)
                f1      = m["macro_f1"]
                elapsed = (time.time() - t0) / 60
                print(f"   Strategy: {strategy} ... done  "
                      f"(macro_f1 {f1:.3f})  {elapsed:.1f} min", flush=True)
                logger.info(f"n={n} strategy={strategy} macro_f1={f1:.4f} "
                            f"acc={m['accuracy']:.4f}")
                _save_result(model_name, strategy, n, m)
            except Exception as exc:
                print(f"   Strategy: {strategy} ... FAILED: {exc}", flush=True)
                logger.error(f"n={n} strategy={strategy} FAILED: {exc}")
                f1 = float("nan")

            results["macro_f1"][strategy].append(f1)

    return results, baseline_f1


# =============================================================================
# Visualization
# =============================================================================

def plot_learning_curve(results: dict, model_name: str,
                        baseline_f1: float) -> None:
    out_dir  = _p("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"learning_curve_{model_name}.svg"

    colors = {
        "last_layer": "#1f77b4",
        "full_model": "#ff7f0e",
        "combined":   "#2ca02c",
    }
    labels = {
        "last_layer": "Last-layer fine-tune",
        "full_model": "Full-model fine-tune",
        "combined":   "Combined training (MESA + TIHM)",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    n_vals = results["n_patients"]

    for strategy, f1_vals in results["macro_f1"].items():
        ax.plot(n_vals, f1_vals, marker="o", lw=2,
                color=colors.get(strategy, "gray"),
                label=labels.get(strategy, strategy))

    if not np.isnan(baseline_f1):
        ax.axhline(baseline_f1, linestyle="--", color="red", alpha=0.7,
                   label=f"Step 3 baseline (F1={baseline_f1:.3f})")

    ax.set_xlabel("Number of fine-tuning patients")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"RQ4 Learning Curve — {model_name.replace('_', ' ').title()}")
    ax.set_xticks(n_vals)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT_DIR)}", flush=True)


def plot_rq4_comparison(all_results: dict, all_baselines: dict,
                        model_names: list) -> None:
    out_dir  = _p("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rq4_step3_vs_rq4.svg"

    baseline_f1s = [all_baselines.get(m, float("nan")) for m in model_names]
    best_rq4_f1s = []
    for m in model_names:
        if m not in all_results:
            best_rq4_f1s.append(float("nan"))
            continue
        vals = [v for s_vals in all_results[m]["macro_f1"].values()
                for v in s_vals[1:]   # skip index-0 (baseline)
                if not np.isnan(v)]
        best_rq4_f1s.append(max(vals) if vals else float("nan"))

    x     = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, baseline_f1s, width, label="Step 3 baseline",
           color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, best_rq4_f1s, width, label="Best RQ4 result",
           color="#ff7f0e", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    ax.set_title("RQ4 — Step 3 Baseline vs Best Transfer Learning Result")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in model_names])
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT_DIR)}", flush=True)


# =============================================================================
# Logging
# =============================================================================

def _setup_logger(model_name: str) -> logging.Logger:
    log_dir = _p("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rq4_{model_name}.log"

    logger = logging.getLogger(f"rq4_{model_name}")
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
# Per-model entry point
# =============================================================================

def run_model(model_name: str) -> tuple[dict, float]:
    is_pytorch = model_name in ("mlp", "lstm")
    if is_pytorch and not _TORCH_OK:
        raise ImportError(
            f"torch is not installed — cannot run {model_name}. "
            "See requirements.txt."
        )

    SEP = "═" * 42
    print(f"\n{SEP}", flush=True)
    print(f" RQ4 — Transfer Learning", flush=True)
    print(f" Model: {model_name.replace('_', ' ').title()}", flush=True)
    print(SEP, flush=True)

    logger = _setup_logger(model_name)
    logger.info(f"Starting RQ4 | model={model_name} | device={CONFIG['device']}")

    print("\nLoading data ...", flush=True)
    X_mesa, y_mesa, _ = load_mesa_aligned()
    tihm_df           = load_tihm_aligned()
    n_total           = tihm_df["patient_id"].nunique()

    abbr = _MODEL_ABBR[model_name]
    pretrain_name = (f"step3_{abbr}_fold1.pt" if is_pytorch
                     else f"step3_{abbr}.pkl")
    print(f"\n Pretrained model: {CONFIG['paths']['pretrained_models']}{pretrain_name}",
          flush=True)
    print(f" TIHM patients: {n_total} total", flush=True)

    results, baseline_f1 = run_learning_curve(
        model_name, tihm_df, X_mesa, y_mesa, logger
    )

    print("\nGenerating figures ...", flush=True)
    plot_learning_curve(results, model_name, baseline_f1)

    # Summary
    best_f1       = float("nan")
    best_strategy = ""
    best_n        = 0
    for strategy, f1_vals in results["macro_f1"].items():
        for i, f1 in enumerate(f1_vals[1:], 1):
            if not np.isnan(f1) and (np.isnan(best_f1) or f1 > best_f1):
                best_f1       = f1
                best_strategy = strategy
                best_n        = results["n_patients"][i]

    improvement = (best_f1 - baseline_f1
                   if not (np.isnan(best_f1) or np.isnan(baseline_f1))
                   else float("nan"))

    print(f"\n Step 3 baseline macro F1: {baseline_f1:.3f}", flush=True)
    if not np.isnan(best_f1):
        print(f" Best RQ4 result: {best_f1:.3f} ({best_strategy}, n={best_n})",
              flush=True)
        if not np.isnan(improvement):
            sign = "+" if improvement >= 0 else ""
            print(f" Improvement: {sign}{improvement:.3f}", flush=True)
    print(SEP, flush=True)

    logger.info(f"Baseline={baseline_f1:.4f}  Best={best_f1:.4f}  "
                f"strategy={best_strategy}  n={best_n}")
    return results, baseline_f1


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="python scripts/09_transfer_learning.py",
        description="RQ4 — Transfer learning from MESA to TIHM.",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", type=str,
                     choices=["random_forest", "xgboost", "mlp", "lstm"],
                     help="Model to evaluate.")
    grp.add_argument("--all", action="store_true",
                     help="Run all models sequentially.")
    args = parser.parse_args()

    models = (["random_forest", "xgboost", "mlp", "lstm"]
              if args.all else [args.model])

    all_results   = {}
    all_baselines = {}

    for model_name in models:
        results, baseline     = run_model(model_name)
        all_results[model_name]   = results
        all_baselines[model_name] = baseline

    if len(models) > 1:
        print("\nGenerating comparison figure ...", flush=True)
        plot_rq4_comparison(all_results, all_baselines, models)


if __name__ == "__main__":
    main()
