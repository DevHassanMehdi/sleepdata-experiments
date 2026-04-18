"""
scripts/04_extract_tihm_features.py
====================================
Extract features from TIHM CSV files (Sleep, Demographics, Physiology).

Produces two outputs per patient:
  Aligned (17 features) — matches MESA aligned feature set for cross-dataset use
  Full    (18 features) — adds body_temp_mean for within-TIHM experiments

Steps:
  1. Load and validate Sleep, Demographics, Physiology CSVs
  2. Split each patient's sleep data into calendar-night sessions.
     Keep only nighttime epochs (20:00–23:59 or 00:00–10:59).
     Trim leading/trailing AWAKE; discard sessions < 30 epochs.
  3. Compute 17 aligned features per epoch
  4. Append body_temp_mean (patient-level scalar) for full output
  5. Save per-patient CSVs + combined CSVs
  6. Print final summary

Aligned feature columns (17) — identical ordering to mesa_aligned_*.csv:
  hr_mean, hr_median,
  rr_mean,
  snore_pct,
  hr_lag1-3, rr_lag1-3, snore_lag1,
  hr_rolling_mean_5, hr_rolling_std_5, rr_rolling_mean_5, rr_rolling_std_5,
  age_group, sex

Spread statistics (hr_std/min/max, rr_std/min/max, snore_std) are excluded
because TIHM provides only one scalar value per minute epoch.

TIHM-only feature column (1):
  body_temp_mean

Usage:
  python scripts/04_extract_tihm_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

SLEEP_CSV  = ROOT_DIR / "data/tihm/Sleep.csv"
DEMO_CSV   = ROOT_DIR / "data/tihm/Demographics.csv"
PHYSIO_CSV = ROOT_DIR / "data/tihm/Physiology.csv"

OUT_ALIGNED_DIR = ROOT_DIR / "outputs/features/tihm_aligned"
OUT_FULL_DIR    = ROOT_DIR / "outputs/features/tihm_full"
OUT_ALIGNED_ALL = ROOT_DIR / "outputs/features/tihm_aligned_all.csv"
OUT_FULL_ALL    = ROOT_DIR / "outputs/features/tihm_full_all.csv"

MIN_SESSION_EPOCHS  = 30   # discard sessions shorter than this after trimming
NIGHT_HOURS_EVENING = 20   # 8 PM onwards counts as "that night"
NIGHT_HOURS_MORNING = 10   # up to 10 AM counts as "that night"

VALID_STATES = {"AWAKE", "LIGHT", "DEEP", "REM"}

# 17 aligned feature columns — must match MESA's FEATURE_COLS order exactly
ALIGNED_COLS = [
    "hr_mean", "hr_median",
    "rr_mean",
    "snore_pct",
    "hr_lag1", "hr_lag2", "hr_lag3",
    "rr_lag1", "rr_lag2", "rr_lag3",
    "snore_lag1",
    "hr_rolling_mean_5", "hr_rolling_std_5",
    "rr_rolling_mean_5", "rr_rolling_std_5",
    "age_group", "sex",
]
assert len(ALIGNED_COLS) == 17

TIHM_ONLY_COLS = ["body_temp_mean"]

META_COLS = ["patient_id", "session_id", "epoch_index"]


# =============================================================================
# Step 1 — Load and validate files
# =============================================================================

def load_all_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Sleep, Demographics, Physiology CSVs and print one-line summaries."""
    print("Loading TIHM data files ...", flush=True)

    sleep  = pd.read_csv(SLEEP_CSV,  parse_dates=["date"])
    demo   = pd.read_csv(DEMO_CSV)
    physio = pd.read_csv(PHYSIO_CSV, parse_dates=["date"])

    for name, df in [("Sleep", sleep), ("Demographics", demo), ("Physiology", physio)]:
        print(f"  {name:<14} {len(df):>8,} rows   cols: {df.columns.tolist()}",
              flush=True)

    unexpected = set(sleep["state"].unique()) - VALID_STATES
    if unexpected:
        print(f"  [WARN] Unexpected state values in Sleep.csv: {sorted(unexpected)}",
              flush=True)
    else:
        print(f"  Sleep.csv states OK: {sorted(sleep['state'].unique())}", flush=True)

    print(flush=True)
    return sleep, demo, physio


# =============================================================================
# Step 2 helpers — session splitting and trimming
# =============================================================================

def _split_sessions(patient_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split a patient's sleep DataFrame into one session per calendar night.

    Only nighttime epochs are kept (hour >= NIGHT_HOURS_EVENING or <= NIGHT_HOURS_MORNING).
    Each epoch is assigned a session_date:
      • hour >= NIGHT_HOURS_EVENING → session_date = that epoch's date
      • hour <= NIGHT_HOURS_MORNING  → session_date = that epoch's date - 1 day

    Rows are grouped by session_date, yielding one session per night.
    """
    df    = patient_df.sort_values("date").reset_index(drop=True)
    hours = df["date"].dt.hour

    night_mask = (hours >= NIGHT_HOURS_EVENING) | (hours <= NIGHT_HOURS_MORNING)
    df = df[night_mask].reset_index(drop=True)
    if df.empty:
        return []

    hours     = df["date"].dt.hour
    date_only = df["date"].dt.normalize()

    session_date = date_only.where(
        hours >= NIGHT_HOURS_EVENING,
        date_only - pd.Timedelta(days=1),
    )
    df = df.copy()
    df["_session_date"] = session_date

    sessions = []
    for _, grp in df.groupby("_session_date", sort=True):
        sessions.append(grp.drop(columns=["_session_date"]).reset_index(drop=True))
    return sessions


def _trim_session(session_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Remove leading and trailing AWAKE epochs from a session.
    Returns None if fewer than MIN_SESSION_EPOCHS clean epochs remain.
    """
    non_awake = session_df.index[session_df["state"] != "AWAKE"]
    if len(non_awake) == 0:
        return None
    trimmed = session_df.loc[non_awake[0]: non_awake[-1]].reset_index(drop=True)
    return trimmed if len(trimmed) >= MIN_SESSION_EPOCHS else None


# =============================================================================
# Step 3 — Aligned feature computation
# =============================================================================

def _compute_aligned_features(
    session_df: pd.DataFrame,
    patient_id: str,
    session_id: int,
    age_group: float,
    sex: float,
) -> pd.DataFrame:
    """
    Build the 24-column aligned feature DataFrame for one session.

    hr_std, hr_min, hr_max, rr_std, rr_min, rr_max, snore_std are NaN because
    TIHM provides only one physiological value per 1-minute epoch — there is no
    within-epoch signal to compute spread from.  These columns are kept so the
    schema matches MESA exactly.
    """
    df = session_df.copy()

    # Signal stats (4) — TIHM provides one scalar per epoch; hr_median == hr_mean
    # because there is no within-epoch distribution to take a median of.
    df["hr_mean"]   = df["heart_rate"]
    df["hr_median"] = df["heart_rate"]
    df["rr_mean"]   = df["respiratory_rate"]
    df["snore_pct"] = df["snoring"].astype(float)

    # Lag features (7) — computed within session, NaN at session start
    df["hr_lag1"]    = df["hr_mean"].shift(1)
    df["hr_lag2"]    = df["hr_mean"].shift(2)
    df["hr_lag3"]    = df["hr_mean"].shift(3)
    df["rr_lag1"]    = df["rr_mean"].shift(1)
    df["rr_lag2"]    = df["rr_mean"].shift(2)
    df["rr_lag3"]    = df["rr_mean"].shift(3)
    df["snore_lag1"] = df["snore_pct"].shift(1)

    # Rolling window features (4)
    df["hr_rolling_mean_5"] = df["hr_mean"].rolling(5, min_periods=1).mean()
    df["hr_rolling_std_5"]  = df["hr_mean"].rolling(5, min_periods=2).std()
    df["rr_rolling_mean_5"] = df["rr_mean"].rolling(5, min_periods=1).mean()
    df["rr_rolling_std_5"]  = df["rr_mean"].rolling(5, min_periods=2).std()

    # Static patient features (2)
    df["age_group"] = age_group
    df["sex"]       = sex

    # Metadata
    df["patient_id"]  = patient_id
    df["session_id"]  = session_id
    df["epoch_index"] = range(len(df))
    df["label"]       = df["state"]

    return df[META_COLS + ALIGNED_COLS + ["label"]].reset_index(drop=True)


# =============================================================================
# Age / sex encoding
# =============================================================================

# TIHM age brackets numbered from 2 to match MESA encoding
# (MESA: (50,60]=1, (60,70]=2, (70,80]=3, (80,90]=4)
_AGE_MAP = {
    "(60, 70]":  1,
    "(70, 80]":  2,
    "(80, 90]":  3,
    "(90, 110]": 4,
}
_SEX_MAP = {"male": 0, "female": 1}


def _encode_demographics(demo_row: pd.Series) -> tuple[float, float]:
    age_group = float(_AGE_MAP.get(str(demo_row["age"]).strip(), np.nan))
    sex       = float(_SEX_MAP.get(str(demo_row["sex"]).strip().lower(), np.nan))
    return age_group, sex


# =============================================================================
# Per-patient pipeline
# =============================================================================

def process_patient(
    patient_id: str,
    sleep_df: pd.DataFrame,
    demo_row: pd.Series,
    body_temp_mean: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all sessions for one patient.

    Returns (aligned_all_sessions, full_all_sessions).
    aligned: 24 features + metadata + label
    full:    aligned + body_temp_mean
    """
    age_group, sex = _encode_demographics(demo_row)
    sessions       = _split_sessions(sleep_df)

    aligned_parts = []
    full_parts    = []
    session_num   = 0

    for raw_session in sessions:
        trimmed = _trim_session(raw_session)
        if trimmed is None:
            continue
        session_num += 1

        aligned = _compute_aligned_features(
            trimmed, patient_id, session_num, age_group, sex
        )
        aligned_parts.append(aligned)

        full = aligned.copy()
        full["body_temp_mean"] = body_temp_mean
        full_parts.append(full)

    if not aligned_parts:
        return pd.DataFrame(), pd.DataFrame()

    return (pd.concat(aligned_parts, ignore_index=True),
            pd.concat(full_parts,    ignore_index=True))


# =============================================================================
# Main
# =============================================================================

def main():
    sleep, demo, physio = load_all_files()

    # Pre-compute per-patient body temperature mean from Physiology.csv
    body_temp_df  = physio[physio["device_type"] == "Body Temperature"]
    body_temp_map = body_temp_df.groupby("patient_id")["value"].mean().to_dict()

    demo_index = demo.set_index("patient_id")

    patients   = sorted(sleep["patient_id"].unique())
    n_patients = len(patients)

    OUT_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FULL_DIR.mkdir(parents=True, exist_ok=True)

    all_aligned = []
    all_full    = []
    total_sessions = 0
    total_epochs   = 0

    print(f"Processing {n_patients} patients ...\n", flush=True)

    for idx, pid in enumerate(patients, start=1):
        sleep_df  = sleep[sleep["patient_id"] == pid].copy()
        demo_row  = demo_index.loc[pid] if pid in demo_index.index else pd.Series()
        body_temp = body_temp_map.get(pid, np.nan)

        aligned_df, full_df = process_patient(pid, sleep_df, demo_row, body_temp)

        if aligned_df.empty:
            print(f"[{idx:02d}/{n_patients}] {pid} — no valid sessions — skipped",
                  flush=True)
            continue

        n_sess   = aligned_df["session_id"].nunique()
        n_epochs = len(aligned_df)
        total_sessions += n_sess
        total_epochs   += n_epochs

        aligned_df.to_csv(OUT_ALIGNED_DIR / f"tihm_aligned_{pid}.csv", index=False)
        full_df.to_csv(   OUT_FULL_DIR    / f"tihm_full_{pid}.csv",    index=False)

        all_aligned.append(aligned_df)
        all_full.append(full_df)

        print(f"[{idx:02d}/{n_patients}] {pid}"
              f" — {n_sess} session{'s' if n_sess != 1 else ''}"
              f" — {n_epochs:,} epochs — saved", flush=True)

    if not all_aligned:
        print("No valid data produced.", flush=True)
        return

    combined_aligned = pd.concat(all_aligned, ignore_index=True)
    combined_full    = pd.concat(all_full,    ignore_index=True)

    combined_aligned.to_csv(OUT_ALIGNED_ALL, index=False)
    combined_full.to_csv(   OUT_FULL_ALL,    index=False)

    # ---- Summary ----
    dist    = combined_aligned["label"].value_counts()
    n_total = len(combined_aligned)
    n_pts   = combined_aligned["patient_id"].nunique()

    avg_sess  = total_sessions / n_pts        if n_pts > 0        else 0
    avg_epoch = total_epochs   / total_sessions if total_sessions > 0 else 0

    n_with_temp = combined_full["body_temp_mean"].notna().groupby(
        combined_full["patient_id"]).any().sum()

    SEP = "\u2550" * 34
    print(f"\n{SEP}", flush=True)
    print(f" TIHM Feature Extraction Complete", flush=True)
    print(SEP, flush=True)
    print(f" Patients processed  : {n_pts}", flush=True)
    print(f" Total sessions      : {total_sessions}", flush=True)
    print(f" Total epochs        : {total_epochs:,}", flush=True)
    print(f" Avg sessions/patient: {avg_sess:.1f}", flush=True)
    print(f" Avg epochs/session  : {avg_epoch:.0f}", flush=True)
    print(f"\n Label distribution:", flush=True)
    for lbl in ["AWAKE", "LIGHT", "DEEP", "REM"]:
        c = dist.get(lbl, 0)
        if c:
            print(f"   {lbl:<8}  {c:>8,}  {c / n_total * 100:5.1f}%", flush=True)
    print(f"\n TIHM-only features:", flush=True)
    print(f"   body_temp_mean : {n_with_temp}/{n_pts} patients have data", flush=True)
    print(f"\n Output schemas:", flush=True)
    print(f"   tihm_aligned_*.csv : {len(META_COLS + ALIGNED_COLS) + 1} cols "
          f"(meta + {len(ALIGNED_COLS)} features + label)", flush=True)
    print(f"   tihm_full_*.csv    : {len(META_COLS + ALIGNED_COLS + TIHM_ONLY_COLS) + 1} cols "
          f"(meta + {len(ALIGNED_COLS)} features + body_temp_mean + label)", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    main()
