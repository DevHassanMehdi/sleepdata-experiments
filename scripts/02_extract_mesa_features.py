"""
scripts/02_extract_mesa_features.py
=====================================
Step 1: Parse NSRR annotation XML → extract and map sleep stage labels
Step 2: Combine 30-second PSG epochs into 1-minute epochs (to match TIHM),
        then trim leading/trailing AWAKE epochs to the actual sleep period.
Step 3: Open the EDF with MNE, list channels, and confirm signal slicing
        for the full PSG channel set.
Step 4: Extract TSFEL features per epoch for the full PSG channel set:
          • Full PSG  → outputs/features/full/mesa_features_full_{sid}.csv

Usage:
  python scripts/02_extract_mesa_features.py                    # subject 0001
  python scripts/02_extract_mesa_features.py --subject 0004
  python scripts/02_extract_mesa_features.py --subjects 350
"""

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

ANNOT_DIR   = ROOT_DIR / "data/mesa/annotations"
EDF_DIR     = ROOT_DIR / "data/mesa/edf"
OUTPUT_DIR  = ROOT_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "mesa_edf_inspection.txt"

# ---------------------------------------------------------------------------
# Channel-specific target sampling rates for TSFEL feature extraction.
# All MESA channels are recorded at 256 Hz. Higher-frequency channels
# (EEG, EMG, EKG, Pleth, Snore) are downsampled to 64 Hz; lower-frequency
# physiological signals (HR, respiratory) are downsampled to 32 Hz.
# resample_poly applies an anti-aliasing filter automatically.
# ---------------------------------------------------------------------------
CHANNEL_TARGET_FS = {
    "EEG1":  64,
    "EEG2":  64,
    "EEG3":  64,
    "EOG-L": 64,
    "EOG-R": 64,
    "EMG":   64,
    "EKG":   64,
    "Pleth": 64,
    "Snore": 64,
    "HR":    32,
    "Thor":  32,
    "Abdo":  32,
    "Flow":  32,
    "SpO2":  32,
}

# Downsample ratios from 256 Hz (resample_poly up/down pairs)
_DS_RATIOS = {64: (1, 4), 32: (1, 8)}

# ---------------------------------------------------------------------------
# Channel sets
# ---------------------------------------------------------------------------
FULL_PSG_CHANNELS = ["EKG", "EOG-L", "EOG-R", "EMG", "EEG1", "EEG2", "EEG3",
                     "Thor", "Abdo", "Flow", "Snore", "SpO2", "HR", "Pleth"]

# ---------------------------------------------------------------------------
# Stage label → 4-class mapping
# ---------------------------------------------------------------------------
STAGE_MAP = {
    # Wake
    "Wake":          "AWAKE",
    "W":             "AWAKE",
    "0":             "AWAKE",
    "wake":          "AWAKE",
    # Light (N1 + N2)
    "N1":            "LIGHT",
    "Stage 1 sleep": "LIGHT",
    "1":             "LIGHT",
    "N2":            "LIGHT",
    "Stage 2 sleep": "LIGHT",
    "2":             "LIGHT",
    # Deep (N3 + N4)
    "N3":            "DEEP",
    "Stage 3 sleep": "DEEP",
    "3":             "DEEP",
    "Stage 4 sleep": "DEEP",
    "4":             "DEEP",
    # REM
    "REM":           "REM",
    "R":             "REM",
    "5":             "REM",
    "REM sleep":     "REM",
}


# =============================================================================
# Step 1 — Parse annotation XML
# =============================================================================

def parse_annotations(xml_path: Path, out: StringIO):
    def p(line=""):
        out.write(line + "\n")

    p("=" * 70)
    p(f"  STEP 1 — Annotation XML: {xml_path.name}")
    p("=" * 70)

    if not xml_path.exists():
        p(f"[ERROR] File not found: {xml_path}")
        return []

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    all_events = list(root.iter(f"{ns}ScoredEvent"))
    p(f"\n  Total ScoredEvent elements in file: {len(all_events)}")
    p(f"\n  Raw XML of first 5 events:")
    p("-" * 70)
    for event in all_events[:5]:
        p(ET.tostring(event, encoding="unicode").strip())
        p()
    p("-" * 70)

    epochs = []
    for event in all_events:
        etype_el   = event.find(f"{ns}EventType")
        concept_el = event.find(f"{ns}EventConcept")
        start_el   = event.find(f"{ns}Start")
        dur_el     = event.find(f"{ns}Duration")

        if etype_el is None or concept_el is None:
            continue
        etype = (etype_el.text or "").strip()
        if "Stages" not in etype and "stage" not in etype.lower():
            continue

        # EventConcept is often "Stage N2|Stage 2 sleep" — take the part before |
        raw_label = (concept_el.text or "").strip().split("|")[0].strip()

        try:
            start    = float(start_el.text) if start_el is not None else 0.0
            duration = float(dur_el.text)   if dur_el   is not None else 30.0
        except (ValueError, AttributeError):
            continue

        epochs.append({
            "start":    start,
            "duration": duration,
            "raw":      raw_label,
        })

    p(f"\n  Stage epochs extracted: {len(epochs)}")

    if not epochs:
        p("  [WARN] No stage epochs found — check EventType values in the raw XML above.")
        return []

    unmapped = set()
    for ep in epochs:
        raw = ep["raw"]
        mapped = STAGE_MAP.get(raw)
        if mapped is None:
            for key, val in STAGE_MAP.items():
                if raw.lower().endswith(key.lower()) or key.lower() in raw.lower():
                    mapped = val
                    break
        if mapped is None:
            unmapped.add(raw)
            mapped = "UNKNOWN"
        ep["label"] = mapped

    if unmapped:
        p(f"\n  [WARN] Unmapped stage labels (add to STAGE_MAP): {sorted(unmapped)}")

    dist = Counter(ep["label"] for ep in epochs)
    total = sum(dist.values())
    p(f"\n  Mapped label distribution ({total} epochs × 30s):")
    for label in ["AWAKE", "LIGHT", "DEEP", "REM", "UNKNOWN"]:
        count = dist.get(label, 0)
        if count == 0:
            continue
        pct  = count / total * 100
        mins = count * 30 / 60
        p(f"    {label:<10}  {count:>5}  ({pct:5.1f}%)  {mins:6.1f} min")

    return epochs


# =============================================================================
# Step 2 — Build continuous timeline, combine into 1-minute epochs, trim
# =============================================================================

def build_continuous_30s_epochs(scored_events: list) -> list:
    """
    The XML contains sparse scored events that may span multiple 30-second
    windows (duration > 30s) or have gaps between them.

    This function:
      1. Expands each event into individual 30-second slots in a lookup dict.
      2. Generates a continuous sequence of 30-second windows from the first
         scored event to the last.
      3. Fills any gap windows using the most recent prior label (forward-fill).

    Returns a gapless list of 30-second epoch dicts with keys:
      start, duration, label, filled (True if forward-filled, False if scored)
    """
    events = sorted(scored_events, key=lambda x: x["start"])

    first_start = events[0]["start"]
    last_end    = max(ev["start"] + ev["duration"] for ev in events)

    # Build a dict: rounded_start_time → label
    # Events with duration > 30s are expanded into multiple 30s slots.
    label_at: dict[float, str] = {}
    for ev in events:
        n_slots = max(1, round(ev["duration"] / 30))
        for j in range(n_slots):
            slot_t = round(ev["start"] + j * 30.0, 3)
            label_at[slot_t] = ev["label"]

    windows = []
    last_label = events[0]["label"]
    t = first_start

    while t < last_end - 0.01:
        key = round(t, 3)
        if key in label_at:
            label  = label_at[key]
            filled = False
            last_label = label
        else:
            label  = last_label   # forward-fill gap
            filled = True

        windows.append({
            "start":    t,
            "duration": 30.0,
            "label":    label,
            "filled":   filled,
        })
        t = round(t + 30.0, 3)

    return windows


def trim_to_sleep_period(epochs_1min: list, out: StringIO) -> list:
    """Remove leading and trailing AWAKE epochs outside the actual sleep period."""
    def p(line=""):
        out.write(line + "\n")

    if not epochs_1min:
        return epochs_1min

    first_sleep = next(
        (i for i, ep in enumerate(epochs_1min) if ep["label"] != "AWAKE"), None
    )
    last_sleep = next(
        (i for i in range(len(epochs_1min) - 1, -1, -1)
         if epochs_1min[i]["label"] != "AWAKE"), None
    )

    if first_sleep is None:
        p("\n  [WARN] All epochs are AWAKE — no sleep period found, keeping full timeline.")
        return epochs_1min

    trimmed = epochs_1min[first_sleep : last_sleep + 1]
    n_removed_lead  = first_sleep
    n_removed_trail = len(epochs_1min) - last_sleep - 1

    p(f"\n  Sleep period trim:")
    p(f"    Leading AWAKE epochs removed  : {n_removed_lead}")
    p(f"    Trailing AWAKE epochs removed : {n_removed_trail}")
    p(f"    Epochs after trim             : {len(trimmed)}")

    dist  = Counter(ep["label"] for ep in trimmed)
    total = sum(dist.values())
    p(f"\n  Trimmed label distribution ({total} epochs × 60s):")
    for label in ["AWAKE", "LIGHT", "DEEP", "REM"]:
        count = dist.get(label, 0)
        if count == 0:
            continue
        pct  = count / total * 100
        p(f"    {label:<10}  {count:>5}  ({pct:5.1f}%)  {count:6.1f} min")

    return trimmed


def combine_to_one_minute(scored_events: list, out: StringIO):
    """
    Returns (epochs_1min, n_mixed) where n_mixed is the count of
    mixed-label pairs that were dropped.
    """
    def p(line=""):
        out.write(line + "\n")

    p("\n" + "=" * 70)
    p("  STEP 2 — Continuous timeline → 1-minute epochs → trim to sleep period")
    p("=" * 70)

    if not scored_events:
        p("  No epochs to combine.")
        return [], 0

    windows_30s = build_continuous_30s_epochs(scored_events)
    n_filled = sum(1 for w in windows_30s if w["filled"])

    p(f"\n  Scored events from XML         : {len(scored_events)}")
    p(f"  Continuous 30s windows created : {len(windows_30s)}")
    p(f"  Forward-filled gap windows     : {n_filled}")
    p(f"  Directly scored windows        : {len(windows_30s) - n_filled}")

    # Pair consecutive 30s windows into 1-minute epochs.
    # Drop any pair where the two 30-second epochs have different labels —
    # keeping only clean same-label pairs ensures label integrity.
    n_pairs_total = len(windows_30s) // 2
    epochs_1min   = []
    n_mixed       = 0
    i = 0
    while i < len(windows_30s) - 1:
        w_a = windows_30s[i]
        w_b = windows_30s[i + 1]

        if w_a["label"] != w_b["label"]:
            n_mixed += 1          # drop — labels disagree
        else:
            epochs_1min.append({
                "start":    w_a["start"],
                "duration": 60.0,
                "label":    w_a["label"],
                "label_a":  w_a["label"],
                "label_b":  w_b["label"],
            })
        i += 2

    n_unpaired = len(windows_30s) % 2
    n_kept     = len(epochs_1min)
    mixed_pct  = n_mixed / n_pairs_total * 100 if n_pairs_total else 0.0
    kept_pct   = n_kept  / n_pairs_total * 100 if n_pairs_total else 0.0

    p(f"\n  30-second epochs input       : {len(windows_30s)}")
    p(f"  1-minute pairs formed        : {n_pairs_total}"
      f"  (+{n_unpaired} unpaired, dropped)")
    p(f"  Mixed-label pairs dropped    : {n_mixed}"
      f"  ({mixed_pct:.1f}%)  — excluded to ensure label integrity")
    p(f"  Clean epochs kept            : {n_kept}"
      f"  ({kept_pct:.1f}%)")

    if not epochs_1min:
        p("  [WARN] No clean epochs remain after dropping mixed pairs.")
        return [], n_mixed

    dist  = Counter(ep["label"] for ep in epochs_1min)
    total = sum(dist.values())
    p(f"\n  Clean 1-minute label distribution ({total} epochs × 60s):")
    for label in ["AWAKE", "LIGHT", "DEEP", "REM"]:
        count = dist.get(label, 0)
        if count == 0:
            continue
        pct = count / total * 100
        p(f"    {label:<10}  {count:>5}  ({pct:5.1f}%)  {count:6.1f} min")

    p(f"\n  First 10 clean epochs:")
    p(f"    {'Start (s)':>10}  {'Epoch A':<10}  {'Epoch B':<10}  {'→ Label'}")
    p("    " + "-" * 50)
    for ep in epochs_1min[:10]:
        p(f"    {ep['start']:>10.0f}  {ep['label_a']:<10}  {ep['label_b']:<10}  {ep['label']}")

    epochs_1min = trim_to_sleep_period(epochs_1min, out)

    return epochs_1min, n_mixed


# =============================================================================
# Step 3 — Read EDF with MNE, confirm channel slicing
# =============================================================================

def _match_channels(target_names: list, edf_channels: list) -> dict:
    """Case-insensitive substring match: target → first matching EDF channel name."""
    ch_lower = {ch.lower(): ch for ch in edf_channels}
    matched = {}
    for target in target_names:
        for lower_name, edf_name in ch_lower.items():
            if target.lower() in lower_name:
                matched[target] = edf_name
                break
    return matched


def inspect_edf(edf_path: Path, epochs_1min: list, out: StringIO):
    def p(line=""):
        out.write(line + "\n")

    p("\n" + "=" * 70)
    p(f"  STEP 3 — EDF file: {edf_path.name}")
    p("=" * 70)

    if not edf_path.exists():
        p(f"[ERROR] File not found: {edf_path}")
        return None

    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        p("[ERROR] mne not installed — run: pip install mne")
        return None

    raw      = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    channels = raw.info["ch_names"]
    sfreq    = raw.info["sfreq"]
    duration_s   = raw.n_times / sfreq
    duration_min = duration_s / 60

    p(f"\n  Sampling frequency : {sfreq:.1f} Hz")
    p(f"  Total duration     : {duration_min:.1f} min  ({duration_s:.0f} s)")
    p(f"  Number of channels : {len(channels)}")

    channel_sfreqs = {}
    try:
        extras = raw._raw_extras[0]
        for i, ch in enumerate(channels):
            native = extras["n_samps"][i] / extras["record_length"]
            channel_sfreqs[ch] = round(native, 2)
    except Exception:
        channel_sfreqs = {ch: sfreq for ch in channels}

    p(f"\n  Channel list:")
    p(f"    {'#':<4}  {'Channel name':<30}  {'Fs (Hz)':>8}")
    p("    " + "-" * 46)
    for idx, ch in enumerate(channels, 1):
        fs = channel_sfreqs.get(ch, sfreq)
        p(f"    {idx:<4}  {ch:<30}  {fs:>8.1f}")

    by_freq: dict[float, list] = {}
    for ch, fs in channel_sfreqs.items():
        by_freq.setdefault(fs, []).append(ch)

    p(f"\n  Unique sampling frequencies:")
    for fs, chs in sorted(by_freq.items(), reverse=True):
        suffix = " ..." if len(chs) > 6 else ""
        p(f"    {fs:>8.1f} Hz  →  {len(chs):>2} channel(s): {', '.join(chs[:6])}{suffix}")

    matched = _match_channels(FULL_PSG_CHANNELS, channels)

    p(f"\n  Channel matching (full PSG set):")
    p(f"    {'Target':<12}  {'EDF channel':<30}")
    p("    " + "-" * 44)
    for target in FULL_PSG_CHANNELS:
        status = matched.get(target, "[NOT FOUND]")
        p(f"    {target:<12}  {status:<30}")

    if not epochs_1min:
        p("\n  [WARN] No 1-minute epochs — skipping slice confirmation.")
        return raw

    first_epoch = epochs_1min[0]
    t_start     = first_epoch["start"]
    t_stop      = t_start + 60.0
    start_idx, stop_idx = raw.time_as_index([t_start, t_stop])

    p(f"\n  Slice confirmation — first epoch: {t_start:.0f}s – {t_stop:.0f}s  "
      f"(label: {first_epoch['label']})")
    p(f"    {'Target':<12}  {'EDF channel':<30}  {'Shape'}")
    p("    " + "-" * 60)

    for target in FULL_PSG_CHANNELS:
        if target not in matched:
            p(f"    {target:<12}  {'—':<30}  [NOT FOUND]")
            continue
        edf_ch  = matched[target]
        ch_idx  = channels.index(edf_ch)
        data, _ = raw[ch_idx, start_idx:stop_idx]
        p(f"    {target:<12}  {edf_ch:<30}  {data.shape}")

    return raw


# =============================================================================
# Step 4 — Full TSFEL feature extraction
# =============================================================================

def _resample_channel(signal: np.ndarray, target: str) -> tuple[np.ndarray, int]:
    """
    Downsample a 256 Hz signal to the channel-specific target rate.
    Uses resample_poly which applies an anti-aliasing filter automatically.
    Returns (resampled_signal, target_fs).
    """
    tgt_fs = CHANNEL_TARGET_FS.get(target, 32)
    up, dn = _DS_RATIOS[tgt_fs]
    return resample_poly(signal, up, dn).astype(np.float32), tgt_fs


def _run_tsfel(tsfel, cfg, signal: np.ndarray, fs: int, target: str) -> dict:
    """Run TSFEL on a signal and prefix column names with the channel target name."""
    feats = tsfel.time_series_features_extractor(cfg, signal, fs=fs, verbose=0)
    feats.columns = [f"{target}__{c}" for c in feats.columns]
    return feats.iloc[0].to_dict()


FEAT_FULL_DIR = OUTPUT_DIR / "features" / "full"


def extract_features(raw, epochs_1min: list, sid: str, out: StringIO) -> dict | None:
    """
    Run TSFEL feature extraction for one subject.

    Parameters
    ----------
    raw        : MNE Raw object (preloaded)
    epochs_1min: trimmed 1-minute epoch list
    sid        : zero-padded subject ID string, e.g. "0001"
    out        : StringIO buffer for detailed log output

    Returns
    -------
    dict with keys: elapsed, n_epochs, label_dist, full_path
    or None on fatal error.
    """
    def p(line=""):
        out.write(line + "\n")

    p("\n" + "=" * 70)
    p("  STEP 4 — TSFEL feature extraction (channel-specific sampling rates)")
    p("=" * 70)

    try:
        import tsfel
    except ImportError:
        p("[ERROR] tsfel not installed — run: pip install tsfel")
        return None

    channels = raw.info["ch_names"]
    orig_fs  = int(raw.info["sfreq"])   # 256 Hz for all MESA channels

    full_matched = _match_channels(FULL_PSG_CHANNELS, channels)
    found_full   = [t for t in FULL_PSG_CHANNELS if t in full_matched]

    if not found_full:
        p("  [ERROR] No target channels found — aborting.")
        return None

    p(f"\n  {'Channel':<10}  {'EDF name':<30}  {'Orig Hz':>7}  {'Target Hz':>9}"
      f"  {'Orig samp':>9}  {'Resamp samp':>11}")
    p("  " + "─" * 76)
    for target in found_full:
        edf_ch = full_matched[target]
        tgt_fs = CHANNEL_TARGET_FS.get(target, 32)
        p(f"  {target:<10}  {edf_ch:<30}  {orig_fs:>7}  {tgt_fs:>9}"
          f"  {orig_fs * 60:>9}  {tgt_fs * 60:>11}")

    cfg      = tsfel.get_features_by_domain()
    t0       = time.time()
    rows_full = []
    n_epochs  = len(epochs_1min)
    p(f"\n  Processing {n_epochs} epochs ...")

    for ep_idx, epoch in enumerate(epochs_1min):
        t_start = epoch["start"]
        t_stop  = t_start + 60.0
        si, ei  = raw.time_as_index([t_start, t_stop])
        label   = epoch["label"]

        row = {}
        for target in found_full:
            ch_idx  = channels.index(full_matched[target])
            data, _ = raw[ch_idx, si:ei]
            sig, fs = _resample_channel(data[0], target)
            row.update(_run_tsfel(tsfel, cfg, sig, fs, target))
        row["label"] = label
        rows_full.append(row)

        if (ep_idx + 1) % 50 == 0 or (ep_idx + 1) == n_epochs:
            p(f"    epoch {ep_idx + 1:>4}/{n_epochs}  ({time.time() - t0:.1f}s elapsed)")

    elapsed = time.time() - t0

    FEAT_FULL_DIR.mkdir(parents=True, exist_ok=True)
    full_path  = None
    label_dist = Counter()

    if rows_full:
        df = pd.DataFrame(rows_full)
        df = df[[c for c in df.columns if c != "label"] + ["label"]]
        full_path  = FEAT_FULL_DIR / f"mesa_features_full_{sid}.csv"
        df.to_csv(full_path, index=False)
        label_dist = Counter(df["label"])

        p(f"\n  Saved: {full_path}")
        p(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        p(f"  Time : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
        p(f"  Label distribution:")
        total = len(df)
        for lbl in ["AWAKE", "LIGHT", "DEEP", "REM"]:
            count = label_dist.get(lbl, 0)
            if count:
                p(f"    {lbl:<10}  {count:>5}  ({count / total * 100:5.1f}%)")

    return {
        "elapsed":    elapsed,
        "n_epochs":   n_epochs,
        "label_dist": label_dist,
        "full_path":  full_path,
    }


BATCH_SUMMARY_CSV = OUTPUT_DIR / "features" / "mesa_batch_summary.csv"
RESUME_MIN_BYTES  = 10 * 1024   # 10 KB — files smaller than this are re-processed


# =============================================================================
# Per-subject pipeline
# =============================================================================

def process_subject(sid: str, save_log: bool = False) -> dict:
    """
    Run the full pipeline (steps 1–4) for one subject.

    Returns a dict with keys:
      sid, status ('processed'|'skipped'|'missing'|'exists'),
      n_epochs, n_awake, n_light, n_deep, n_rem, n_mixed,
      processing_time_seconds, label_dist, full_path
    """
    xml_path = ANNOT_DIR / f"mesa-sleep-{sid}-nsrr.xml"
    edf_path = EDF_DIR   / f"mesa-sleep-{sid}.edf"
    full_out = FEAT_FULL_DIR / f"mesa_features_full_{sid}.csv"

    def _empty(status: str) -> dict:
        return {"sid": sid, "status": status,
                "n_epochs": 0, "n_awake": 0, "n_light": 0, "n_deep": 0, "n_rem": 0,
                "n_mixed": 0,
                "processing_time_seconds": 0.0, "label_dist": Counter(),
                "full_path": None}

    # Missing input files
    if not xml_path.exists() or not edf_path.exists():
        return _empty("missing")

    # Resume: full output exists and is large enough
    full_ok = full_out.exists() and full_out.stat().st_size >= RESUME_MIN_BYTES
    if full_ok:
        try:
            labels = pd.read_csv(full_out, usecols=["label"])["label"]
            dist   = Counter(labels)
        except Exception:
            dist = Counter()
        n = sum(dist.values())
        return {"sid": sid, "status": "exists",
                "n_epochs": n,
                "n_awake":  dist.get("AWAKE", 0),
                "n_light":  dist.get("LIGHT", 0),
                "n_deep":   dist.get("DEEP",  0),
                "n_rem":    dist.get("REM",   0),
                "n_mixed":  0,
                "processing_time_seconds": 0.0,
                "label_dist": dist,
                "full_path": full_out}

    buf = StringIO()

    epochs_30s           = parse_annotations(xml_path, buf)
    epochs_1min, n_mixed = combine_to_one_minute(epochs_30s, buf)

    if not epochs_1min:
        return _empty("skipped")

    raw = inspect_edf(edf_path, epochs_1min, buf)
    if raw is None:
        return _empty("skipped")

    feat_result = extract_features(raw, epochs_1min, sid, buf)
    if feat_result is None:
        return _empty("skipped")

    if save_log:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())

    dist = feat_result["label_dist"]
    return {"sid": sid, "status": "processed",
            "n_epochs": feat_result["n_epochs"],
            "n_awake":  dist.get("AWAKE", 0),
            "n_light":  dist.get("LIGHT", 0),
            "n_deep":   dist.get("DEEP",  0),
            "n_rem":    dist.get("REM",   0),
            "n_mixed":  n_mixed,
            "processing_time_seconds": feat_result["elapsed"],
            "label_dist": dist,
            "full_path": feat_result["full_path"]}


# =============================================================================
# Helpers
# =============================================================================

def _fmt_time(secs: float) -> str:
    if secs >= 3600:
        return f"{secs / 3600:.1f} hr"
    if secs >= 60:
        return f"{secs / 60:.1f} min"
    return f"{secs:.0f}s"


def _fmt_dist(label_dist, n_epochs: int) -> str:
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
        prog="python scripts/02_extract_mesa_features.py",
        description="Extract TSFEL features from MESA PSG data.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--subject", metavar="ID",
                      help="Process a single subject by 4-digit ID.")
    mode.add_argument("--subjects", metavar="N", type=int,
                      help="Process the first N subjects sequentially (batch mode).")
    args = parser.parse_args()

    # ------------------------------------------------------------------ single
    if args.subjects is None:
        sid    = (args.subject or "0001").zfill(4)
        result = process_subject(sid, save_log=True)
        status = result["status"]
        dist   = result["label_dist"]

        print(f"Subject {sid}", flush=True)
        if status == "missing":
            print(f"  Error  : EDF or XML not found", flush=True)
        elif status == "skipped":
            print(f"  Error  : no clean epochs (see {OUTPUT_FILE})", flush=True)
        else:
            n = result["n_epochs"]
            tag = "  (already done)" if status == "exists" else ""
            print(f"  Epochs : {n}{tag}"
                  f"  (AWAKE {dist.get('AWAKE', 0)}"
                  f" / LIGHT {dist.get('LIGHT', 0)}"
                  f" / DEEP {dist.get('DEEP', 0)}"
                  f" / REM {dist.get('REM', 0)})", flush=True)
            if status == "processed":
                print(f"  Dropped: {result['n_mixed']} mixed-label pairs", flush=True)
                print(f"  Time   : {result['processing_time_seconds'] / 60:.1f} min", flush=True)
            print(f"  Saved  : {result['full_path'].relative_to(ROOT_DIR)}", flush=True)
        return

    # ------------------------------------------------------------------ batch
    n_target      = args.subjects
    t_batch_start = time.time()
    total_dist:   Counter = Counter()
    meta_rows:    list    = []
    n_processed   = 0
    n_resumed     = 0
    n_skipped     = 0
    total_epochs  = 0
    total_proc_s  = 0.0

    consecutive_missing = 0

    for i in range(1, 10000):
        done = n_processed + n_resumed
        if done >= n_target:
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
        result = process_subject(sid, save_log=False)
        status = result["status"]

        meta_rows.append({
            "subject_id":              sid,
            "n_epochs":                result["n_epochs"],
            "n_awake":                 result["n_awake"],
            "n_light":                 result["n_light"],
            "n_deep":                  result["n_deep"],
            "n_rem":                   result["n_rem"],
            "processing_time_seconds": result["processing_time_seconds"],
            "status":                  status,
        })

        if status in ("missing", "skipped"):
            n_skipped += 1
            print(f"Skipping {sid} — no EDF found", flush=True)
            continue

        n_ep = result["n_epochs"]
        total_epochs += n_ep
        total_dist   += result["label_dist"]

        if status == "exists":
            n_resumed += 1
            print(f"Skipping {sid} — already done", flush=True)
        else:
            n_processed  += 1
            total_proc_s += result["processing_time_seconds"]
            dist_str      = _fmt_dist(result["label_dist"], n_ep)
            t_min         = result["processing_time_seconds"] / 60
            print(f"Done — {n_ep} epochs ({dist_str}) — {t_min:.1f} min", flush=True)

        done = n_processed + n_resumed
        if done > 0 and done % 25 == 0:
            elapsed = time.time() - t_batch_start
            print(f"── Progress: {done}/{n_target} subjects done"
                  f" — {elapsed / 3600:.1f} hr elapsed ──", flush=True)

    # ---- Save metadata CSV ----
    BATCH_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta_rows).to_csv(BATCH_SUMMARY_CSV, index=False)

    # ---- Final summary ----
    elapsed_total = time.time() - t_batch_start
    grand         = sum(total_dist.values())
    avg_s         = (total_proc_s / n_processed) if n_processed > 0 else 0.0

    SEP = "\u2550" * 34
    print(f"\n{SEP}", flush=True)
    print(f" MESA Feature Extraction Complete", flush=True)
    print(SEP, flush=True)
    print(f" Processed    : {n_processed} subjects", flush=True)
    print(f" Already done : {n_resumed}", flush=True)
    print(f" Skipped      : {n_skipped}  (no EDF found)", flush=True)
    print(f" Total epochs : {total_epochs:,}", flush=True)
    print(f" Total time   : {_fmt_time(elapsed_total)}", flush=True)
    print(f" Avg/subject  : {_fmt_time(avg_s)}", flush=True)
    if grand > 0:
        print(f"\n Label distribution (all subjects):", flush=True)
        for lbl in ["AWAKE", "LIGHT", "DEEP", "REM"]:
            c = total_dist.get(lbl, 0)
            if c:
                print(f"   {lbl:<8}  {c:>8,}  ({c / grand * 100:5.1f}%)", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    main()
