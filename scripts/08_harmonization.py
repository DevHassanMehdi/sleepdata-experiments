"""
scripts/08_harmonization.py
===========================
Apply two harmonization methods to MESA and TIHM aligned feature datasets.

Methods:
  1. Z-score  : StandardScaler fit on combined MESA+TIHM, applied to both
  2. ComBat   : Subject-level batch-effect correction via neuroCombat,
                covariates = age_group + sex

Outputs (outputs/features/harmonized/):
  mesa_zscore.csv, tihm_zscore.csv
  mesa_combat.csv, tihm_combat.csv

Figures:
  outputs/figures/harmonization_effect.svg

Usage:
  python scripts/08_harmonization.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

ALIGNED_DIR = ROOT_DIR / "outputs/features/aligned"
TIHM_CSV    = ROOT_DIR / "outputs/features/tihm_aligned_all.csv"
OUT_DIR     = ROOT_DIR / "outputs/features/harmonized"
FIG_PATH    = ROOT_DIR / "outputs/figures/harmonization_effect.svg"

LABEL_MAP = {"AWAKE": 0, "LIGHT": 1, "DEEP": 2, "REM": 3}

FEATURE_COLS = [
    "hr_mean", "hr_median", "rr_mean", "snore_pct",
    "hr_lag1", "hr_lag2", "hr_lag3",
    "rr_lag1", "rr_lag2", "rr_lag3", "snore_lag1",
    "hr_rolling_mean_5", "hr_rolling_std_5",
    "rr_rolling_mean_5", "rr_rolling_std_5",
    "age_group", "sex",
]
assert len(FEATURE_COLS) == 17

# ComBat harmonises signal features; age_group + sex are treated as covariates
COMBAT_FEATURE_COLS = [c for c in FEATURE_COLS if c not in ("age_group", "sex")]


# =============================================================================
# Step 2 — MMD helper
# =============================================================================

def compute_mmd(X_source: np.ndarray, X_target: np.ndarray,
                gamma: float | None = None) -> tuple[float, float]:
    """
    Maximum Mean Discrepancy with RBF kernel.

    gamma=None uses the median heuristic: gamma = 1 / (2 * median_dist^2).
    Returns (mmd_value, gamma_used).
    """
    from sklearn.metrics.pairwise import rbf_kernel
    if gamma is None:
        from sklearn.metrics import pairwise_distances
        sample = np.vstack([X_source[:1000], X_target[:1000]])
        dists  = pairwise_distances(sample)
        median_dist = np.median(dists[dists > 0])
        gamma = float(1.0 / (2 * median_dist ** 2))
    XX  = rbf_kernel(X_source, X_source, gamma)
    YY  = rbf_kernel(X_target, X_target, gamma)
    XY  = rbf_kernel(X_source, X_target, gamma)
    mmd = float(XX.mean() + YY.mean() - 2 * XY.mean())
    return mmd, gamma


def mmd_subsample(mesa_vals: np.ndarray, tihm_vals: np.ndarray,
                  n: int = 5000, gamma: float | None = None,
                  seed: int = 42) -> tuple[float, float]:
    """Returns (mmd, gamma_used)."""
    rng   = np.random.default_rng(seed)
    idx_m = rng.choice(len(mesa_vals), min(n, len(mesa_vals)), replace=False)
    idx_t = rng.choice(len(tihm_vals), min(n, len(tihm_vals)), replace=False)
    return compute_mmd(mesa_vals[idx_m], tihm_vals[idx_t], gamma)


# =============================================================================
# Step 1 — Load and clean data
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading data ...", flush=True)

    # ── MESA aligned ──────────────────────────────────────────────────────────
    files = sorted(ALIGNED_DIR.glob("mesa_aligned_*.csv"))
    if not files:
        raise FileNotFoundError(f"No MESA aligned CSVs found in {ALIGNED_DIR}")

    frames = []
    for f in files:
        sid = f.stem.rsplit("_", 1)[-1]
        tmp = pd.read_csv(f)
        tmp["subject_id"] = sid
        frames.append(tmp)
    mesa_df = pd.concat(frames, ignore_index=True)

    # Encode labels — use is_integer_dtype because pandas StringDtype != object
    if not pd.api.types.is_integer_dtype(mesa_df["label"]):
        mesa_df["label"] = mesa_df["label"].map(LABEL_MAP)
    mesa_df["label"] = mesa_df["label"].astype(int)

    before = len(mesa_df)
    mesa_df = mesa_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if before - len(mesa_df):
        print(f"  [info] MESA: dropped {before - len(mesa_df):,} NaN-feature rows",
              flush=True)
    mesa_df["batch"] = 0

    # ── TIHM aligned ─────────────────────────────────────────────────────────
    if not TIHM_CSV.exists():
        raise FileNotFoundError(f"TIHM aligned CSV not found: {TIHM_CSV}")
    tihm_df = pd.read_csv(TIHM_CSV)

    if not pd.api.types.is_integer_dtype(tihm_df["label"]):
        tihm_df["label"] = tihm_df["label"].map(LABEL_MAP)
    tihm_df["label"] = tihm_df["label"].astype(int)

    before = len(tihm_df)
    tihm_df = tihm_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if before - len(tihm_df):
        print(f"  [info] TIHM: dropped {before - len(tihm_df):,} NaN-feature rows",
              flush=True)
    tihm_df["batch"] = 1

    # ── Summary ───────────────────────────────────────────────────────────────
    n_m_subj = mesa_df["subject_id"].nunique()
    n_t_subj = tihm_df["patient_id"].nunique()
    print(f"  MESA    : {len(mesa_df):,} epochs, {n_m_subj} subjects", flush=True)
    print(f"  TIHM    : {len(tihm_df):,} epochs, {n_t_subj} patients", flush=True)
    print(f"  Combined: {len(mesa_df) + len(tihm_df):,} epochs\n", flush=True)

    print("  Per-feature distribution differences:", flush=True)
    hdr = (f"  {'Feature':<25} {'MESA mean':>10} {'TIHM mean':>10}"
           f" {'MESA std':>10} {'TIHM std':>10}")
    print(hdr, flush=True)
    print("  " + "-" * 67, flush=True)
    for feat in FEATURE_COLS:
        m_mean = mesa_df[feat].mean()
        t_mean = tihm_df[feat].mean()
        m_std  = mesa_df[feat].std()
        t_std  = tihm_df[feat].std()
        print(f"  {feat:<25} {m_mean:>10.3f} {t_mean:>10.3f}"
              f" {m_std:>10.3f} {t_std:>10.3f}", flush=True)
    print(flush=True)

    return mesa_df, tihm_df


# =============================================================================
# Step 3 — Z-score harmonization
# =============================================================================

def harmonize_zscore(
    mesa_df: pd.DataFrame,
    tihm_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Fit on combined MESA + TIHM data so neither dataset defines the reference
    # distribution in isolation.
    combined_features = np.vstack([
        mesa_df[FEATURE_COLS].values,
        tihm_df[FEATURE_COLS].values,
    ])
    scaler = StandardScaler()
    scaler.fit(combined_features)

    mesa_out = mesa_df.copy()
    tihm_out = tihm_df.copy()
    mesa_out[FEATURE_COLS] = scaler.transform(mesa_df[FEATURE_COLS].values)
    tihm_out[FEATURE_COLS] = scaler.transform(tihm_df[FEATURE_COLS].values)

    return mesa_out, tihm_out


# =============================================================================
# Step 4 — ComBat harmonization
# =============================================================================

def harmonize_combat(
    mesa_df: pd.DataFrame,
    tihm_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from neuroCombat import neuroCombat

    # ── Subject-level means ───────────────────────────────────────────────────
    mesa_feat_means  = mesa_df.groupby("subject_id")[COMBAT_FEATURE_COLS].mean()
    tihm_feat_means  = tihm_df.groupby("patient_id")[COMBAT_FEATURE_COLS].mean()
    mesa_covar_means = mesa_df.groupby("subject_id")[["age_group", "sex"]].mean()
    tihm_covar_means = tihm_df.groupby("patient_id")[["age_group", "sex"]].mean()

    # Prefix keys to guarantee uniqueness across datasets
    mesa_feat_means.index  = ["mesa_" + str(i) for i in mesa_feat_means.index]
    tihm_feat_means.index  = ["tihm_" + str(i) for i in tihm_feat_means.index]
    mesa_covar_means.index = ["mesa_" + str(i) for i in mesa_covar_means.index]
    tihm_covar_means.index = ["tihm_" + str(i) for i in tihm_covar_means.index]

    mesa_feat_means["batch"] = 0
    tihm_feat_means["batch"] = 1
    mesa_feat_means["age_group"] = mesa_covar_means["age_group"]
    mesa_feat_means["sex"]       = mesa_covar_means["sex"].round().astype(int)
    tihm_feat_means["age_group"] = tihm_covar_means["age_group"]
    tihm_feat_means["sex"]       = tihm_covar_means["sex"].round().astype(int)

    combined_means = pd.concat([mesa_feat_means, tihm_feat_means])

    # Drop subjects with missing covariates; fill any remaining NaN in features
    combined_means = combined_means.dropna(subset=["age_group", "sex"])
    combined_means[COMBAT_FEATURE_COLS] = (
        combined_means[COMBAT_FEATURE_COLS]
        .fillna(combined_means[COMBAT_FEATURE_COLS].median())
    )

    # ── Run neuroCombat ───────────────────────────────────────────────────────
    # neuroCombat expects shape (n_features, n_subjects)
    data   = combined_means[COMBAT_FEATURE_COLS].T.values.astype(float)
    covars = combined_means[["batch", "age_group", "sex"]].reset_index(drop=True)

    combat_result = neuroCombat(
        dat=data,
        covars=covars,
        batch_col="batch",
        categorical_cols=["sex"],
        continuous_cols=["age_group"],
    )

    harmonized_means = pd.DataFrame(
        combat_result["data"].T,
        columns=COMBAT_FEATURE_COLS,
        index=combined_means.index,
    )

    # ── Apply epoch-level correction ──────────────────────────────────────────
    orig_means = combined_means[COMBAT_FEATURE_COLS]
    correction = harmonized_means - orig_means

    mesa_out = mesa_df.copy()
    tihm_out = tihm_df.copy()

    for sid in mesa_df["subject_id"].unique():
        key  = "mesa_" + str(sid)
        mask = mesa_out["subject_id"] == sid
        if key in correction.index:
            corr = correction.loc[key, COMBAT_FEATURE_COLS].values
            mesa_out.loc[mask, COMBAT_FEATURE_COLS] = (
                mesa_out.loc[mask, COMBAT_FEATURE_COLS].values + corr
            )

    for pid in tihm_df["patient_id"].unique():
        key  = "tihm_" + str(pid)
        mask = tihm_out["patient_id"] == pid
        if key in correction.index:
            corr = correction.loc[key, COMBAT_FEATURE_COLS].values
            tihm_out.loc[mask, COMBAT_FEATURE_COLS] = (
                tihm_out.loc[mask, COMBAT_FEATURE_COLS].values + corr
            )

    return mesa_out, tihm_out


# =============================================================================
# Step 5 — Visualization
# =============================================================================

def _kde_plot(ax, mesa_vals: np.ndarray, tihm_vals: np.ndarray,
              title: str, xlabel: str) -> None:
    """KDE overlay of MESA (blue) vs TIHM (purple) for one feature."""
    for vals, color, label in [
        (mesa_vals, "#1f77b4", "MESA"),
        (tihm_vals, "#9467bd", "TIHM"),
    ]:
        clean = vals[np.isfinite(vals)]
        if len(clean) < 2:
            continue
        lo, hi = np.percentile(clean, 1), np.percentile(clean, 99)
        if lo >= hi:
            lo, hi = clean.min(), clean.max()
        xs = np.linspace(lo, hi, 300)
        try:
            kde = gaussian_kde(clean, bw_method="scott")
            ax.plot(xs, kde(xs), color=color, lw=1.5, label=label)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
        except Exception:
            pass
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)


def plot_harmonization_effect(
    raw_m: pd.DataFrame, raw_t: pd.DataFrame,
    zsc_m: pd.DataFrame, zsc_t: pd.DataFrame,
    cbt_m: pd.DataFrame, cbt_t: pd.DataFrame,
    mmd_before: float, mmd_zscore: float, mmd_combat: float,
) -> None:
    plot_feats = ["hr_mean", "rr_mean", "snore_pct"]
    labels_row = [
        f"Before harmonization  (MMD={mmd_before:.4f})",
        f"After z-score         (MMD={mmd_zscore:.4f})",
        f"After ComBat          (MMD={mmd_combat:.4f})",
    ]
    datasets = [
        (raw_m, raw_t),
        (zsc_m, zsc_t),
        (cbt_m, cbt_t),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle("Feature distribution alignment: MESA vs TIHM", fontsize=11)

    for row_idx, ((mesa_data, tihm_data), row_lbl) in enumerate(
        zip(datasets, labels_row)
    ):
        for col_idx, feat in enumerate(plot_feats):
            ax = axes[row_idx, col_idx]
            title = f"{row_lbl}" if col_idx == 0 else ""
            _kde_plot(
                ax,
                mesa_data[feat].values,
                tihm_data[feat].values,
                title=title,
                xlabel=feat,
            )

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, format="svg")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "outputs/figures").mkdir(parents=True, exist_ok=True)

    # ── Step 1: load ──────────────────────────────────────────────────────────
    mesa_raw, tihm_raw = load_data()

    # ── Step 2: MMD before ────────────────────────────────────────────────────
    print("Computing MMD before harmonization ...", flush=True)
    mmd_before, gamma_used = mmd_subsample(
        mesa_raw[FEATURE_COLS].values,
        tihm_raw[FEATURE_COLS].values,
    )
    print(f"  gamma (median heuristic): {gamma_used:.6g}", flush=True)
    print(f"  MMD before: {mmd_before:.4f}\n", flush=True)

    # ── Step 3: Z-score ───────────────────────────────────────────────────────
    print("Applying z-score harmonization (joint MESA+TIHM scaler) ...", flush=True)
    mesa_zsc, tihm_zsc = harmonize_zscore(mesa_raw, tihm_raw)
    mmd_zscore, _ = mmd_subsample(
        mesa_zsc[FEATURE_COLS].values,
        tihm_zsc[FEATURE_COLS].values,
        gamma=gamma_used,   # reuse the same gamma for comparable values
    )
    print(f"  MMD after z-score: {mmd_zscore:.4f}", flush=True)

    _label_enc = {"AWAKE": 0, "LIGHT": 1, "DEEP": 2, "REM": 3}
    if not pd.api.types.is_integer_dtype(mesa_zsc["label"]):
        mesa_zsc["label"] = mesa_zsc["label"].map(_label_enc)
    mesa_zsc["label"] = mesa_zsc["label"].astype(int)
    if not pd.api.types.is_integer_dtype(tihm_zsc["label"]):
        tihm_zsc["label"] = tihm_zsc["label"].map(_label_enc)
    tihm_zsc["label"] = tihm_zsc["label"].astype(int)
    mesa_zsc.to_csv(OUT_DIR / "mesa_zscore.csv", index=False)
    tihm_zsc.to_csv(OUT_DIR / "tihm_zscore.csv", index=False)
    print("  Saved: mesa_zscore.csv, tihm_zscore.csv\n", flush=True)

    # ── Step 4: ComBat ────────────────────────────────────────────────────────
    print("Applying ComBat harmonization ...", flush=True)
    try:
        mesa_cbt, tihm_cbt = harmonize_combat(mesa_raw, tihm_raw)
        mmd_combat, _ = mmd_subsample(
            mesa_cbt[FEATURE_COLS].values,
            tihm_cbt[FEATURE_COLS].values,
            gamma=gamma_used,
        )
        print(f"  MMD after ComBat: {mmd_combat:.4f}", flush=True)
        _label_enc = {"AWAKE": 0, "LIGHT": 1, "DEEP": 2, "REM": 3}
        if not pd.api.types.is_integer_dtype(mesa_cbt["label"]):
            mesa_cbt["label"] = mesa_cbt["label"].map(_label_enc)
        mesa_cbt["label"] = mesa_cbt["label"].astype(int)
        if not pd.api.types.is_integer_dtype(tihm_cbt["label"]):
            tihm_cbt["label"] = tihm_cbt["label"].map(_label_enc)
        tihm_cbt["label"] = tihm_cbt["label"].astype(int)
        mesa_cbt.to_csv(OUT_DIR / "mesa_combat.csv", index=False)
        tihm_cbt.to_csv(OUT_DIR / "tihm_combat.csv", index=False)
        print("  Saved: mesa_combat.csv, tihm_combat.csv\n", flush=True)
        combat_ok = True
    except Exception as exc:
        print(f"  [warn] ComBat failed: {exc}", flush=True)
        print("  Falling back to z-score for ComBat outputs.", flush=True)
        mesa_cbt, tihm_cbt = mesa_zsc, tihm_zsc
        mmd_combat = mmd_zscore
        combat_ok  = False

    # ── Step 5: visualisation ─────────────────────────────────────────────────
    print("Generating harmonization figure ...", flush=True)
    plot_harmonization_effect(
        mesa_raw, tihm_raw,
        mesa_zsc, tihm_zsc,
        mesa_cbt, tihm_cbt,
        mmd_before, mmd_zscore, mmd_combat,
    )
    print(f"  Saved: {FIG_PATH.relative_to(ROOT_DIR)}\n", flush=True)

    # ── Step 6: summary ───────────────────────────────────────────────────────
    pct_zsc = (mmd_before - mmd_zscore) / mmd_before * 100 if mmd_before > 0 else 0
    pct_cbt = (mmd_before - mmd_combat) / mmd_before * 100 if mmd_before > 0 else 0

    SEP = "═" * 38
    print(f"\n{SEP}", flush=True)
    print(" Harmonization Complete", flush=True)
    print(SEP, flush=True)
    print(f" MMD before harmonization : {mmd_before:.4f}", flush=True)
    print(f" MMD after z-score        : {mmd_zscore:.4f}  ({pct_zsc:.1f}% reduction)",
          flush=True)
    print(f" MMD after ComBat         : {mmd_combat:.4f}  ({pct_cbt:.1f}% reduction)",
          flush=True)
    if not combat_ok:
        print(" [warn] ComBat failed — z-score used as fallback for combat outputs",
              flush=True)
    print(f"\n Saved:", flush=True)
    for name in ["mesa_zscore.csv", "tihm_zscore.csv", "mesa_combat.csv", "tihm_combat.csv"]:
        print(f"   {(OUT_DIR / name).relative_to(ROOT_DIR)}", flush=True)
    print(f"   {FIG_PATH.relative_to(ROOT_DIR)}", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    main()
