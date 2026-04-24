"""
utils/training_utils.py
=======================
PyTorch training loop, metric computation, and sample-weight helpers.
"""

import numpy as np
import torch
import torch.nn as nn


def train_pytorch_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    writer,
    fold: int,
    model_name: str,
) -> tuple[nn.Module, dict]:
    """
    Train a PyTorch model with early stopping and mixed-precision (AMP).

    Returns (best_model, history) where history has keys:
      train_loss, val_loss, val_acc — one value per epoch.
    """
    scaler  = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val_loss    = float("inf")
    best_state       = None
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ── training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(X_batch)
                loss   = criterion(output, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    output = model(X_batch)
                    loss   = criterion(output, y_batch)
                val_loss += loss.item()
                pred      = output.argmax(dim=1)
                correct  += (pred == y_batch).sum().item()
                total    += len(y_batch)

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_acc     = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if writer:
            writer.add_scalar(f"fold{fold}/train_loss", train_loss, epoch)
            writer.add_scalar(f"fold{fold}/val_loss",   val_loss,   epoch)
            writer.add_scalar(f"fold{fold}/val_acc",    val_acc,    epoch)

        # ── early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if scheduler is not None:
            scheduler.step(val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def get_sample_weights(y: np.ndarray, class_weights_dict: dict) -> np.ndarray:
    """Map per-class weight dict to a per-sample weight array (for XGBoost)."""
    return np.array([class_weights_dict[label] for label in y])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    """Compute accuracy, per-class F1, macro F1, ROC-AUC, PR-AUC."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(label_names):
        metrics[f"f1_{name.lower()}"] = float(per_class[i]) if i < len(per_class) else 0.0

    try:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(np.mean([
            average_precision_score((y_true == i).astype(int), y_proba[:, i])
            for i in range(y_proba.shape[1])
        ]))
    except Exception:
        metrics["pr_auc"] = float("nan")

    return metrics
