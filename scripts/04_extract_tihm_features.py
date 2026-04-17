"""
scripts/04_extract_tihm_features.py
=====================================
Extract the same 26 harmonized features from TIHM wearable sleep data.

Input : data/tihm/Sleep.csv
        Columns: patient_id, date, state, heart_rate, respiratory_rate, snoring

Feature set (26 total — mirrors script 03):
  HR features (10)    : mean, std, min, max, median, skewness, kurtosis,
                        iqr, rmssd, cv
  RR features (7)     : mean, std, min, max, median, skewness, cv
  Snoring features (2): presence (bool), fraction (window ratio)
  Composite (7)       : HR_delta, RR_delta, HR_RR_ratio,
                        HR_trend, RR_trend, HR_mad, RR_mad

Window size: ±WINDOW_HALF (default 2) epochs around each target epoch,
giving a 5-minute window of wearable data for computing statistics.
Window shrinks at session boundaries (gaps > GAP_THRESHOLD seconds).
HR_trend / RR_trend use a [0, 1]-normalised time axis to match script 03.

Output: outputs/features/tihm_features.csv
        (patient_id + date + 26 features + label, one row per 1-minute epoch)

Usage:
  python scripts/04_extract_tihm_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from tqdm import tqdm

ROOT_DIR  = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

SLEEP_CSV  = ROOT_DIR / "data/tihm/Sleep.csv"
OUTPUT_CSV = ROOT_DIR / "outputs/features/tihm_features.csv"

WINDOW_HALF     = 2    # epochs before and after → 5-minute window total
GAP_THRESHOLD_S = 300  # seconds; larger gap → new session

FEATURE_NAMES = [
    "HR_mean", "HR_std", "HR_min", "HR_max", "HR_median",
    "HR_skewness", "HR_kurtosis", "HR_iqr", "HR_rmssd", "HR_cv",
    "RR_mean", "RR_std", "RR_min", "RR_max", "RR_median",
    "RR_skewness", "RR_cv",
    "snoring_presence", "snoring_fraction",
    "HR_delta", "RR_delta", "HR_RR_ratio", "HR_trend", "RR_trend",
    "HR_mad", "RR_mad",
]
assert len(FEATURE_NAMES) == 26, f"Expected 26, got {len(FEATURE_NAMES)}"


# =============================================================================
# Feature computation from a window of scalar values
# =============================================================================

def _hr_features(window: np.ndarray) -> dict:
    """Compute 10 HR features from a window of heart_rate values (bpm)."""
    x        = window.astype(np.float64)
    mean_val = float(np.mean(x))
    std_val  = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    q75, q25 = (np.percentile(x, [75, 25]) if len(x) >= 4
                else (float(np.max(x)), float(np.min(x))))
    diffs    = np.diff(x)
    rmssd    = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0

    return {
        "HR_mean":     mean_val,
        "HR_std":      std_val,
        "HR_min":      float(np.min(x)),
        "HR_max":      float(np.max(x)),
        "HR_median":   float(np.median(x)),
        "HR_skewness": float(skew(x))     if len(x) >= 3 else 0.0,
        "HR_kurtosis": float(kurtosis(x)) if len(x) >= 4 else 0.0,
        "HR_iqr":      float(q75 - q25),
        "HR_rmssd":    rmssd,
        "HR_cv":       std_val / mean_val if mean_val != 0 else 0.0,
    }


def _rr_features(window: np.ndarray) -> dict:
    """Compute 7 RR features from a window of respiratory_rate values."""
    x        = window.astype(np.float64)
    mean_val = float(np.mean(x))
    std_val  = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    return {
        "RR_mean":     mean_val,
        "RR_std":      std_val,
        "RR_min":      float(np.min(x)),
        "RR_max":      float(np.max(x)),
        "RR_median":   float(np.median(x)),
        "RR_skewness": float(skew(x)) if len(x) >= 3 else 0.0,
        "RR_cv":       std_val / mean_val if mean_val != 0 else 0.0,
    }


def _composite_features(hr_window: np.ndarray, rr_window: np.ndarray) -> dict:
    """
    Compute 7 composite features.

    HR_trend / RR_trend slopes use a [0, 1]-normalised time axis to match
    the normalisation used in script 03 (MESA) — both yield change-per-epoch.
    """
    x = hr_window.astype(np.float64)
    r = rr_window.astype(np.float64)

    hr_mean = float(np.mean(x))
    rr_mean = float(np.mean(r))

    t_x      = np.linspace(0.0, 1.0, len(x))
    hr_slope = float(np.polyfit(t_x, x, 1)[0]) if len(x) > 1 else 0.0

    t_r      = np.linspace(0.0, 1.0, len(r))
    rr_slope = float(np.polyfit(t_r, r, 1)[0]) if len(r) > 1 else 0.0

    return {
        "HR_delta":    float(np.max(x) - np.min(x)),
        "RR_delta":    float(np.max(r) - np.min(r)),
        "HR_RR_ratio": hr_mean / rr_mean if rr_mean != 0 else np.nan,
        "HR_trend":    hr_slope,
        "RR_trend":    rr_slope,
        "HR_mad":      float(np.mean(np.abs(x - hr_mean))),
        "RR_mad":      float(np.mean(np.abs(r - rr_mean))),
    }


# =============================================================================
# Per-patient processing
# =============================================================================

def _session_ids(dates: pd.Series) -> np.ndarray:
    """
    Assign a monotonically increasing session ID to each row.
    A new session starts when the gap to the previous row exceeds
    GAP_THRESHOLD_S seconds.
    """
    diffs = dates.diff().dt.total_seconds().fillna(0)
    return (diffs > GAP_THRESHOLD_S).cumsum().to_numpy()


def process_patient(df_patient: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 26 aligned features for all epochs of one patient.

    Returns a DataFrame with columns: patient_id, date, <26 features>, label.
    """
    df = df_patient.sort_values("date").reset_index(drop=True)

    hr       = df["heart_rate"].to_numpy(dtype=float)
    rr       = df["respiratory_rate"].to_numpy(dtype=float)
    snoring  = df["snoring"].to_numpy(dtype=bool)
    sessions = _session_ids(df["date"])
    n        = len(df)

    rows = []
    for i in range(n):
        sess = sessions[i]

        # Expand window left/right while staying within the same session
        lo = i
        while lo > 0 and (i - lo) < WINDOW_HALF and sessions[lo - 1] == sess:
            lo -= 1
        hi = i
        while hi < n - 1 and (hi - i) < WINDOW_HALF and sessions[hi + 1] == sess:
            hi += 1

        hr_win = hr[lo: hi + 1]
        rr_win = rr[lo: hi + 1]
        sn_win = snoring[lo: hi + 1]

        row = {}
        row.update(_hr_features(hr_win))
        row.update(_rr_features(rr_win))
        row.update(_composite_features(hr_win, rr_win))
        row["snoring_presence"] = int(snoring[i])
        row["snoring_fraction"] = float(sn_win.mean())
        row["label"]            = df.at[i, "state"]

        rows.append(row)

    result = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
    result.insert(0, "patient_id", df["patient_id"].values)
    result.insert(1, "date",       df["date"].values)
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Reading {SLEEP_CSV.relative_to(ROOT_DIR)} ...")
    df = pd.read_csv(SLEEP_CSV, parse_dates=["date"])

    # Ensure snoring is boolean (may be stored as string "True"/"False")
    if df["snoring"].dtype == object:
        df["snoring"] = df["snoring"].map({"True": True, "False": False}).astype(bool)

    patients = sorted(df["patient_id"].unique())
    print(f"Patients: {len(patients)}   Total epochs: {len(df):,}\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for pid in tqdm(patients, desc="Processing TIHM patients", unit="patient"):
        df_p   = df[df["patient_id"] == pid].copy()
        result = process_patient(df_p)
        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    n_total = len(combined)
    dist    = combined["label"].value_counts()
    SEP = "\u2550" * 38
    print(f"\n{SEP}")
    print(f" TIHM Feature Extraction Complete")
    print(SEP)
    print(f" Patients     : {len(patients)}")
    print(f" Total epochs : {n_total:,}")
    print(f" Features     : {len(FEATURE_NAMES)}")
    print(f"\n Label distribution:")
    for lbl in ["AWAKE", "LIGHT", "DEEP", "REM"]:
        c = dist.get(lbl, 0)
        if c:
            print(f"   {lbl:<8}  {c:>8,}  ({c / n_total * 100:5.1f}%)")
    print(f"\n Saved: {OUTPUT_CSV.relative_to(ROOT_DIR)}")
    print(SEP)


if __name__ == "__main__":
    main()
