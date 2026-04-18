"""
scripts/03_extract_mesa_aligned_features.py
===========================================
Extract 17 aligned features from MESA EDF files for cross-dataset comparison.

Features are designed to match what TIHM wearables can provide.
Spread statistics (hr_std/min/max, rr_std/min/max, snore_std) are excluded
because TIHM only provides one scalar value per minute — keeping them would
create a permanently-NaN column gap between datasets.

Feature groups (17 total):
  Signal stats   ( 4): hr_mean, hr_median, rr_mean, snore_pct
  Lag            ( 7): hr_lag1-3, rr_lag1-3, snore_lag1
  Rolling window ( 4): hr_rolling_mean_5, hr_rolling_std_5,
                        rr_rolling_mean_5, rr_rolling_std_5
  Static patient ( 2): age_group, sex

Epoch pipeline is identical to script 02:
  XML parse → stage map → continuous 30-s timeline → 1-min pairs
  → drop mixed-label pairs → trim leading/trailing AWAKE

Lag and rolling features are NaN where prior epochs do not exist.

Output: outputs/features/aligned/mesa_aligned_{sid}.csv

Usage:
  python scripts/03_extract_mesa_aligned_features.py               # subject 0001
  python scripts/03_extract_mesa_aligned_features.py --subject 0004
  python scripts/03_extract_mesa_aligned_features.py --subjects 300
"""

import argparse
import importlib.util
import sys
import time
from io import StringIO
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample_poly

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

ANNOT_DIR        = ROOT_DIR / "data/mesa/annotations"
EDF_DIR          = ROOT_DIR / "data/mesa/edf"
HARMONIZED_CSV   = ROOT_DIR / "data/mesa/datasets/mesa-sleep-harmonized-dataset-0.8.0.csv"
FEAT_ALIGNED_DIR = ROOT_DIR / "outputs/features/aligned"
RESUME_MIN_BYTES = 10 * 1024   # 10 KB

SNORE_THRESHOLD  = 0.01        # Snore channel amplitude threshold for snore_pct

# Downsample from 256 Hz (all MESA EDF channels share this native rate)
_DS_RATIOS = {64: (1, 4), 32: (1, 8)}
_CH_FS     = {"HR": 32, "Flow": 32, "Snore": 64}

FEATURE_COLS = [
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
assert len(FEATURE_COLS) == 17


# =============================================================================
# Load shared annotation / epoch helpers from script 02
# =============================================================================

def _load_s02():
    spec = importlib.util.spec_from_file_location(
        "mesa_02", Path(__file__).parent / "02_extract_mesa_features.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s02 = _load_s02()
_parse_annotations     = _s02.parse_annotations
_combine_to_1min       = _s02.combine_to_one_minute
_match_channels        = _s02._match_channels


# =============================================================================
# Load patient metadata once at startup
# =============================================================================

def _load_patient_meta() -> pd.DataFrame:
    """
    Load age and sex from the harmonized CSV.
    Index is nsrrid (integer).
    """
    df = pd.read_csv(HARMONIZED_CSV, usecols=["nsrrid", "nsrr_age", "nsrr_sex"])

    # Encode age → age_group: (50,60]=1, (60,70]=2, (70,80]=3, (80,90]=4
    df["age_group"] = pd.cut(
        df["nsrr_age"],
        bins=[50, 60, 70, 80, 90],
        labels=[1, 2, 3, 4],
    ).astype(float)

    # Encode sex: male=0, female=1
    df["sex"] = df["nsrr_sex"].str.lower().map({"male": 0, "female": 1})

    return df.set_index("nsrrid")


# =============================================================================
# Signal extraction helpers
# =============================================================================

def _downsample(signal: np.ndarray, channel: str) -> np.ndarray:
    """Downsample a 256 Hz MESA channel to its target rate."""
    up, dn = _DS_RATIOS[_CH_FS[channel]]
    return resample_poly(signal, up, dn).astype(np.float32)


def estimate_rr_bpm(flow_signal: np.ndarray, fs: int) -> float:
    """
    Estimate mean respiratory rate in breaths per minute from raw Flow channel.

    Flow alternates positive (inhale) and negative (exhale). Taking the
    absolute value turns each breath cycle into a peak; find_peaks counts them.
    Returns rr_mean (breaths/min), or NaN if the signal is empty.
    """
    abs_signal   = np.abs(flow_signal)
    min_distance = int(fs * 1.5)          # max 40 breaths/min → ≥1.5 s apart
    prominence   = np.std(abs_signal) * 0.3

    peaks, _ = find_peaks(abs_signal, distance=min_distance, prominence=prominence)

    duration_minutes = len(flow_signal) / fs / 60
    return float(len(peaks) / duration_minutes) if duration_minutes > 0 else np.nan


def _signal_stats_for_epoch(
    raw,
    channels: list[str],
    matched: dict,
    t_start: float,
) -> dict | None:
    """
    Extract HR, Flow, Snore stats for one 1-minute epoch.

    Returns a dict with 11 keys, or None if any channel slice is empty.
    """
    t_stop       = t_start + 60.0
    si, ei       = raw.time_as_index([t_start, t_stop])

    hr_idx       = channels.index(matched["HR"])
    hr_raw, _    = raw[hr_idx, si:ei]
    hr            = _downsample(hr_raw[0], "HR").astype(np.float64)

    fl_idx       = channels.index(matched["Flow"])
    fl_raw, _    = raw[fl_idx, si:ei]
    flow          = _downsample(fl_raw[0], "Flow").astype(np.float64)

    sn_idx       = channels.index(matched["Snore"])
    sn_raw, _    = raw[sn_idx, si:ei]
    snore         = _downsample(sn_raw[0], "Snore").astype(np.float64)

    if hr.size == 0 or flow.size == 0 or snore.size == 0:
        return None

    rr_mean = estimate_rr_bpm(flow, fs=_CH_FS["Flow"])

    return {
        "hr_mean":   float(np.mean(hr)),
        "hr_median": float(np.median(hr)),
        "rr_mean":   rr_mean,
        "snore_pct": float(np.mean(np.abs(snore) > SNORE_THRESHOLD)),
    }


# =============================================================================
# Lag and rolling window features (computed on the full epoch table)
# =============================================================================

def _add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling-window features in-place.
    Values are NaN at the start of the session where prior epochs are missing.
    """
    df["hr_lag1"]  = df["hr_mean"].shift(1)
    df["hr_lag2"]  = df["hr_mean"].shift(2)
    df["hr_lag3"]  = df["hr_mean"].shift(3)
    df["rr_lag1"]  = df["rr_mean"].shift(1)
    df["rr_lag2"]  = df["rr_mean"].shift(2)
    df["rr_lag3"]  = df["rr_mean"].shift(3)
    df["snore_lag1"] = df["snore_pct"].shift(1)

    roll = df["hr_mean"].rolling(window=5, min_periods=1)
    df["hr_rolling_mean_5"] = roll.mean()
    df["hr_rolling_std_5"]  = df["hr_mean"].rolling(window=5, min_periods=2).std()

    roll_rr = df["rr_mean"].rolling(window=5, min_periods=1)
    df["rr_rolling_mean_5"] = roll_rr.mean()
    df["rr_rolling_std_5"]  = df["rr_mean"].rolling(window=5, min_periods=2).std()

    # Lag features are genuinely missing at the start — keep as NaN
    return df


# =============================================================================
# Per-subject extraction pipeline
# =============================================================================

def extract_subject(sid: str, meta: pd.DataFrame, raw) -> pd.DataFrame | None:
    """
    Build the full aligned feature table for one subject.

    Returns a DataFrame with FEATURE_COLS + ['label'], or None on failure.
    """
    buf = StringIO()
    xml_path = ANNOT_DIR / f"mesa-sleep-{sid}-nsrr.xml"

    epochs_30s          = _parse_annotations(xml_path, buf)
    epochs_1min, _      = _combine_to_1min(epochs_30s, buf)

    if not epochs_1min:
        return None

    channels = raw.info["ch_names"]
    matched  = _match_channels(["HR", "Flow", "Snore"], channels)
    missing  = [ch for ch in ("HR", "Flow", "Snore") if ch not in matched]
    if missing:
        return None

    rows = []
    for epoch in epochs_1min:
        stats = _signal_stats_for_epoch(raw, channels, matched, epoch["start"])
        if stats is None:
            continue
        stats["label"] = epoch["label"]
        rows.append(stats)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = _add_lag_rolling(df)

    # Static patient features
    nsrrid = int(sid)
    if nsrrid in meta.index:
        row_meta       = meta.loc[nsrrid]
        df["age_group"] = row_meta["age_group"]
        df["sex"]       = row_meta["sex"]
    else:
        df["age_group"] = np.nan
        df["sex"]       = np.nan

    return df[FEATURE_COLS + ["label"]]


# =============================================================================
# Per-subject pipeline (with file I/O and resume logic)
# =============================================================================

def process_subject(sid: str, meta: pd.DataFrame) -> dict:
    """
    Full pipeline for one subject: annotations → EDF → features → CSV.

    Returns a result dict with keys: sid, status, n_epochs, elapsed_s, label_dist.
    """
    xml_path = ANNOT_DIR / f"mesa-sleep-{sid}-nsrr.xml"
    edf_path = EDF_DIR   / f"mesa-sleep-{sid}.edf"
    out_path = FEAT_ALIGNED_DIR / f"mesa_aligned_{sid}.csv"

    def _result(status, n_epochs=0, elapsed=0.0, label_dist=None):
        return {"sid": sid, "status": status, "n_epochs": n_epochs,
                "elapsed_s": elapsed, "label_dist": label_dist or {}}

    if not xml_path.exists() or not edf_path.exists():
        return _result("missing")

    if out_path.exists() and out_path.stat().st_size >= RESUME_MIN_BYTES:
        try:
            labels     = pd.read_csv(out_path, usecols=["label"])["label"]
            label_dist = labels.value_counts().to_dict()
            n          = len(labels)
        except Exception:
            label_dist, n = {}, 0
        return _result("exists", n_epochs=n, label_dist=label_dist)

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    t0 = time.time()
    df = extract_subject(sid, meta, raw)
    elapsed = time.time() - t0

    if df is None or df.empty:
        return _result("skipped")

    label_dist = df["label"].value_counts().to_dict()
    FEAT_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return _result("processed", n_epochs=len(df), elapsed=elapsed, label_dist=label_dist)


# =============================================================================
# CLI helpers
# =============================================================================

def _fmt_min(s: float) -> str:
    return f"{s / 60:.1f} min"


def _fmt_dist(label_dist: dict, n_epochs: int) -> str:
    if n_epochs == 0:
        return "no epochs"
    parts = []
    for lbl in ["AWAKE", "LIGHT", "DEEP", "REM"]:
        c   = label_dist.get(lbl, 0)
        pct = round(c / n_epochs * 100)
        parts.append(f"{lbl} {pct}%")
    return " / ".join(parts)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="python scripts/03_extract_mesa_aligned_features.py",
        description="Extract 24 aligned features from MESA PSG data.",
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--subject",  metavar="ID",
                     help="Single subject by 4-digit ID (default: 0001).")
    grp.add_argument("--subjects", metavar="N", type=int,
                     help="Process first N available subjects (batch mode).")
    args = parser.parse_args()

    meta = _load_patient_meta()

    # ------------------------------------------------------------------ single
    if args.subjects is None:
        sid = (args.subject or "0001").zfill(4)
        print(f"Processing {sid} ...", flush=True)
        res = process_subject(sid, meta)

        if res["status"] == "missing":
            print(f"Skipping {sid} — no EDF or XML", flush=True)
        elif res["status"] == "skipped":
            print(f"Skipping {sid} — no clean epochs", flush=True)
        elif res["status"] == "exists":
            print(f"Skipping {sid} — already done", flush=True)
        else:
            n        = res["n_epochs"]
            t        = _fmt_min(res["elapsed_s"])
            dist_str = _fmt_dist(res["label_dist"], n)
            print(f"Done — {n} epochs ({dist_str}) — {t}", flush=True)
            out = FEAT_ALIGNED_DIR / f"mesa_aligned_{sid}.csv"
            print(f"Saved: {out.relative_to(ROOT_DIR)}", flush=True)
        return

    # ------------------------------------------------------------------ batch
    n_target     = args.subjects
    t_start      = time.time()
    n_processed  = 0
    n_skipped    = 0
    total_epochs = 0
    total_dist: dict[str, int] = {}
    consecutive_missing = 0

    for i in range(1, 10_000):
        if n_processed >= n_target:
            break

        sid      = f"{i:04d}"
        xml_path = ANNOT_DIR / f"mesa-sleep-{sid}-nsrr.xml"
        edf_path = EDF_DIR   / f"mesa-sleep-{sid}.edf"

        if not xml_path.exists() and not edf_path.exists():
            consecutive_missing += 1
            if consecutive_missing >= 20:
                break
            continue
        consecutive_missing = 0

        print(f"Processing {sid} ...", flush=True)
        res    = process_subject(sid, meta)
        status = res["status"]

        if status == "missing":
            n_skipped += 1
            print(f"Skipping {sid} — no EDF found", flush=True)
            continue

        if status == "skipped":
            n_skipped += 1
            print(f"Skipping {sid} — no clean epochs", flush=True)
            continue

        n_ep = res["n_epochs"]
        total_epochs += n_ep
        for lbl, cnt in res["label_dist"].items():
            total_dist[lbl] = total_dist.get(lbl, 0) + int(cnt)
        n_processed += 1

        if status == "exists":
            print(f"Skipping {sid} — already done", flush=True)
        else:
            dist_str = _fmt_dist(res["label_dist"], n_ep)
            t_min    = _fmt_min(res["elapsed_s"])
            print(f"Done — {n_ep} epochs ({dist_str}) — {t_min}", flush=True)

        if n_processed > 0 and n_processed % 25 == 0:
            elapsed = time.time() - t_start
            print(f"── Progress: {n_processed}/{n_target} subjects done"
                  f" — {elapsed / 3600:.1f} hr elapsed ──", flush=True)

    elapsed_total = time.time() - t_start
    grand         = sum(total_dist.values())

    SEP = "\u2550" * 38
    print(f"\n{SEP}", flush=True)
    print(f" MESA Aligned Feature Extraction Done", flush=True)
    print(SEP, flush=True)
    print(f" Processed   : {n_processed} subjects", flush=True)
    print(f" Skipped     : {n_skipped}  (no EDF)", flush=True)
    print(f" Total epochs: {total_epochs:,}", flush=True)
    print(f" Total time  : {_fmt_min(elapsed_total)}", flush=True)
    if n_processed > 0:
        print(f" Avg/subject : {_fmt_min(elapsed_total / n_processed)}", flush=True)
    if grand > 0:
        print(f"\n Label distribution:", flush=True)
        for lbl in ["AWAKE", "LIGHT", "DEEP", "REM"]:
            c = total_dist.get(lbl, 0)
            if c:
                print(f"   {lbl:<8}  {c:>8,}  {c / grand * 100:5.1f}%", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    main()
