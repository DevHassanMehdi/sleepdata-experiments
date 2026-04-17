"""
scripts/05_dataset_overview.py
================================
Dataset profiling, statistics, and visualisations for MESA and TIHM.

Produces:
  outputs/overview/dataset_overview_summary.txt
  outputs/overview/01_mesa_subject_coverage.svg
  outputs/overview/02_label_distribution_comparison.svg
  outputs/overview/03_hr_distribution_comparison.svg
  outputs/overview/04_rr_distribution_comparison.svg
  outputs/overview/05_mesa_stage_distribution_per_subject.svg
  outputs/overview/06_tihm_patient_timeline.svg
  outputs/overview/07_feature_alignment_heatmap.svg
  outputs/overview/08_tihm_clinical_events.svg

Usage:
  python scripts/05_dataset_overview.py
"""

import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.logger import get_logger
from utils.plotting import save_figure

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MESA_HARMONIZED = ROOT_DIR / "data/mesa/datasets/mesa-sleep-harmonized-dataset-0.8.0.csv"
MESA_EDF_DIR    = ROOT_DIR / "data/mesa/edf"
MESA_ANNOT_DIR  = ROOT_DIR / "data/mesa/annotations"
FEAT_FULL_DIR   = ROOT_DIR / "outputs/features/full"
TIHM_SLEEP      = ROOT_DIR / "data/tihm/Sleep.csv"
TIHM_DEMO       = ROOT_DIR / "data/tihm/Demographics.csv"
TIHM_PHYSIO     = ROOT_DIR / "data/tihm/Physiology.csv"
TIHM_ACTIVITY   = ROOT_DIR / "data/tihm/Activity.csv"
TIHM_LABELS     = ROOT_DIR / "data/tihm/Labels.csv"
OVERVIEW_DIR    = ROOT_DIR / "outputs/overview"
SUMMARY_FILE    = OVERVIEW_DIR / "dataset_overview_summary.txt"

log = get_logger("05_dataset_overview", "all")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def apply_thesis_style():
    sns.set_theme(style="white", font="sans-serif")
    plt.rcParams.update({
        "figure.figsize":    (12, 6),
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
    })


# ---------------------------------------------------------------------------
# Section 1 — Load data
# ---------------------------------------------------------------------------

def load_data():
    data = {}

    def try_load(key, path, **kwargs):
        if path.exists():
            try:
                data[key] = pd.read_csv(path, **kwargs)
                log.info(f"Loaded {key}: {data[key].shape}")
            except Exception as e:
                log.warning(f"Could not load {key} from {path}: {e}")
        else:
            log.warning(f"Missing file: {path}")

    try_load("mesa_harm", MESA_HARMONIZED, low_memory=False)
    try_load("tihm_sleep", TIHM_SLEEP, low_memory=False)
    try_load("tihm_demo", TIHM_DEMO, low_memory=False)
    try_load("tihm_physio", TIHM_PHYSIO, low_memory=False)
    try_load("tihm_activity", TIHM_ACTIVITY, low_memory=False)
    try_load("tihm_labels", TIHM_LABELS, low_memory=False)

    # Count file system assets
    data["n_edf"]   = len(list(MESA_EDF_DIR.glob("*.edf")))   if MESA_EDF_DIR.exists()   else 0
    data["n_annot"] = len(list(MESA_ANNOT_DIR.glob("*.xml"))) if MESA_ANNOT_DIR.exists() else 0
    data["n_feat"]  = len(list(FEAT_FULL_DIR.glob("*.csv")))  if FEAT_FULL_DIR.exists()  else 0
    data["feat_files"] = sorted(FEAT_FULL_DIR.glob("*.csv"))  if FEAT_FULL_DIR.exists()  else []

    log.info(f"EDF files: {data['n_edf']}, annotation XMLs: {data['n_annot']}, feature CSVs: {data['n_feat']}")
    return data


# ---------------------------------------------------------------------------
# Section 2 — MESA harmonized stats
# ---------------------------------------------------------------------------

def mesa_harmonized_stats(data, lines):
    lines.append("\n" + "=" * 70)
    lines.append("MESA HARMONIZED DATASET")
    lines.append("=" * 70)

    df = data.get("mesa_harm")
    if df is None:
        lines.append("  [MISSING] mesa-sleep-harmonized-dataset-0.8.0.csv not found.")
        return

    lines.append(f"  Subjects (rows)      : {len(df):,}")
    lines.append(f"  EDF files on disk    : {data['n_edf']:,}")
    lines.append(f"  Annotation XMLs      : {data['n_annot']:,}")
    lines.append(f"  Feature CSVs         : {data['n_feat']:,}")

    if "nsrr_age" in df.columns:
        age = df["nsrr_age"].dropna()
        lines.append(f"\n  Age: mean={age.mean():.1f}  std={age.std():.1f}"
                     f"  min={age.min():.0f}  max={age.max():.0f}")

    if "nsrr_sex" in df.columns:
        lines.append(f"\n  Sex distribution:")
        for val, cnt in df["nsrr_sex"].value_counts().items():
            lines.append(f"    {val}: {cnt:,}")

    for col, label in [
        ("nsrr_ahi_hp3u",     "AHI (events/hr)"),
        ("nsrr_tst_f1",       "Total sleep time (min)"),
        ("nsrr_ttleffsp_f1",  "Sleep efficiency (%)"),
    ]:
        if col in df.columns:
            s = df[col].dropna()
            lines.append(f"\n  {label}: mean={s.mean():.1f}  std={s.std():.1f}")

    stage_cols = {
        "nsrr_pctdursp_s1": "N1 %",
        "nsrr_pctdursp_s2": "N2 %",
        "nsrr_pctdursp_s3": "N3 %",
        "nsrr_pctdursp_sr": "REM %",
    }
    lines.append(f"\n  Sleep stage percentages (mean ± std):")
    for col, lbl in stage_cols.items():
        if col in df.columns:
            s = df[col].dropna()
            lines.append(f"    {lbl:<8}: {s.mean():5.1f}% ± {s.std():.1f}%")


# ---------------------------------------------------------------------------
# Section 3 — MESA extracted feature stats
# ---------------------------------------------------------------------------

def mesa_feature_stats(data, lines):
    lines.append("\n" + "=" * 70)
    lines.append("MESA EXTRACTED FEATURES")
    lines.append("=" * 70)

    feat_files = data.get("feat_files", [])
    if not feat_files:
        lines.append("  [MISSING] No feature CSVs found in outputs/features/full/")
        return

    lines.append(f"  Feature CSV files    : {len(feat_files):,}")

    # Load only label column across all subjects
    label_frames = []
    for f in feat_files:
        try:
            label_frames.append(pd.read_csv(f, usecols=["label"]))
        except Exception:
            pass

    if not label_frames:
        lines.append("  [WARN] Could not read label columns from feature CSVs.")
        return

    all_labels = pd.concat(label_frames, ignore_index=True)["label"]
    dist = Counter(all_labels)
    total = len(all_labels)
    mean_epochs = total / len(feat_files)

    lines.append(f"  Total epochs         : {total:,}")
    lines.append(f"  Mean epochs/subject  : {mean_epochs:.0f}")
    lines.append(f"\n  Label distribution:")
    for lbl in ["AWAKE", "LIGHT", "DEEP", "REM"]:
        c = dist.get(lbl, 0)
        lines.append(f"    {lbl:<8}: {c:>7,}  ({c / total * 100:5.1f}%)")

    # Subjects with zero DEEP or zero REM
    zero_deep = zero_rem = 0
    for f in feat_files:
        try:
            labels = pd.read_csv(f, usecols=["label"])["label"]
            lcnt   = Counter(labels)
            if lcnt.get("DEEP", 0) == 0:
                zero_deep += 1
            if lcnt.get("REM", 0) == 0:
                zero_rem += 1
        except Exception:
            pass
    lines.append(f"\n  Subjects with 0 DEEP epochs : {zero_deep}")
    lines.append(f"  Subjects with 0 REM epochs  : {zero_rem}")

    # Store per-subject label series for later plots
    data["mesa_all_labels"] = all_labels
    data["mesa_feat_files"] = feat_files
    data["mesa_label_dist"] = dist
    data["mesa_total_epochs"] = total


# ---------------------------------------------------------------------------
# Section 4 — TIHM stats
# ---------------------------------------------------------------------------

def tihm_stats(data, lines):
    lines.append("\n" + "=" * 70)
    lines.append("TIHM DATASET")
    lines.append("=" * 70)

    sleep = data.get("tihm_sleep")
    if sleep is None:
        lines.append("  [MISSING] TIHM Sleep.csv not found.")
    else:
        # Detect patient ID column
        pid_col = next((c for c in sleep.columns if "patient" in c.lower()), None)
        if pid_col is None and sleep.columns[0] not in ("state", "label", "heart_rate"):
            pid_col = sleep.columns[0]

        if pid_col:
            n_patients = sleep[pid_col].nunique()
            lines.append(f"  Unique patients      : {n_patients}")

        lines.append(f"  Total sleep epochs   : {len(sleep):,}")

        # Label distribution
        label_col = next((c for c in sleep.columns if c.lower() in ("state", "label", "sleep_state")), None)
        if label_col:
            lines.append(f"\n  Sleep state distribution ({label_col}):")
            for val, cnt in sleep[label_col].value_counts().items():
                lines.append(f"    {val}: {cnt:,}  ({cnt / len(sleep) * 100:.1f}%)")
            data["tihm_label_col"] = label_col

        # HR stats
        hr_col = next((c for c in sleep.columns if "heart" in c.lower() or c.lower() == "hr"), None)
        if hr_col:
            hr = sleep[hr_col].dropna()
            lines.append(f"\n  Heart rate ({hr_col}): mean={hr.mean():.1f}  std={hr.std():.1f}"
                         f"  min={hr.min():.0f}  max={hr.max():.0f}")
            data["tihm_hr"] = hr

        # RR stats
        rr_col = next((c for c in sleep.columns if "respir" in c.lower() or "rr" == c.lower()), None)
        if rr_col:
            rr = sleep[rr_col].dropna()
            lines.append(f"  Respiratory rate ({rr_col}): mean={rr.mean():.1f}  std={rr.std():.1f}"
                         f"  min={rr.min():.0f}  max={rr.max():.0f}")
            data["tihm_rr"] = rr

        # Snoring %
        snore_col = next((c for c in sleep.columns if "snor" in c.lower()), None)
        if snore_col:
            snore_rate = sleep[snore_col].mean() * 100
            lines.append(f"  Snoring rate         : {snore_rate:.1f}%")

        data["tihm_sleep_df"] = sleep
        if pid_col:
            data["tihm_pid_col"] = pid_col

    demo = data.get("tihm_demo")
    if demo is not None:
        lines.append(f"\n  Demographics rows    : {len(demo):,}")
        age_col = next((c for c in demo.columns if "age" in c.lower()), None)
        sex_col = next((c for c in demo.columns if "sex" in c.lower() or "gender" in c.lower()), None)
        if age_col:
            lines.append(f"  Age groups ({age_col}):")
            for val, cnt in demo[age_col].value_counts().items():
                lines.append(f"    {val}: {cnt}")
        if sex_col:
            lines.append(f"  Sex ({sex_col}):")
            for val, cnt in demo[sex_col].value_counts().items():
                lines.append(f"    {val}: {cnt}")

    physio = data.get("tihm_physio")
    if physio is not None:
        dev_col = next((c for c in physio.columns if "device" in c.lower()), None)
        if dev_col:
            lines.append(f"\n  Physiology device types ({dev_col}):")
            for val, cnt in physio[dev_col].value_counts().items():
                lines.append(f"    {val}: {cnt}")

    activity = data.get("tihm_activity")
    if activity is not None:
        loc_col = next((c for c in activity.columns if "location" in c.lower()), None)
        if loc_col:
            lines.append(f"\n  Activity events by location (top 10):")
            for val, cnt in activity[loc_col].value_counts().head(10).items():
                lines.append(f"    {val}: {cnt:,}")

    labels_df = data.get("tihm_labels")
    if labels_df is not None:
        type_col = next((c for c in labels_df.columns if c.lower() == "type"), None)
        if type_col:
            lines.append(f"\n  Clinical event types ({type_col}):")
            for val, cnt in labels_df[type_col].value_counts().items():
                lines.append(f"    {val}: {cnt:,}")


# ---------------------------------------------------------------------------
# Section 5 — Plots
# ---------------------------------------------------------------------------

def plot_01_mesa_coverage(data):
    n_harm = len(data["mesa_harm"]) if "mesa_harm" in data else 0
    n_edf  = data["n_edf"]
    n_feat = data["n_feat"]

    fig, ax = plt.subplots()
    labels  = ["Harmonized\nsubjects", "EDF files\non disk", "Feature\nCSVs"]
    values  = [n_harm, n_edf, n_feat]
    colors  = ["#4C72B0", "#55A868", "#C44E52"]
    bars    = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=11)
    ax.set_title("MESA Dataset — Subject Coverage")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values) * 1.15)
    save_figure(fig, OVERVIEW_DIR, "01_mesa_subject_coverage.svg")


def plot_02_label_comparison(data):
    mesa_dist  = data.get("mesa_label_dist", Counter())
    mesa_total = data.get("mesa_total_epochs", 1) or 1

    sleep_df  = data.get("tihm_sleep_df")
    tihm_dist = Counter()
    label_col = data.get("tihm_label_col")
    if sleep_df is not None and label_col:
        tihm_dist = Counter(sleep_df[label_col].dropna())
    tihm_total = sum(tihm_dist.values()) or 1

    # Map TIHM labels to 4-class if possible
    TIHM_MAP = {
        "Wake": "AWAKE", "W": "AWAKE",
        "Light": "LIGHT", "N1": "LIGHT", "N2": "LIGHT",
        "Deep": "DEEP", "N3": "DEEP",
        "REM": "REM", "R": "REM",
    }
    tihm_mapped: Counter = Counter()
    for k, v in tihm_dist.items():
        mapped = TIHM_MAP.get(str(k), str(k))
        tihm_mapped[mapped] += v

    stages  = ["AWAKE", "LIGHT", "DEEP", "REM"]
    x       = np.arange(len(stages))
    width   = 0.35
    mesa_pct  = [mesa_dist.get(s, 0) / mesa_total * 100 for s in stages]
    tihm_pct  = [tihm_mapped.get(s, 0) / (sum(tihm_mapped.values()) or 1) * 100 for s in stages]

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, mesa_pct, width, label="MESA (PSG)", color="#4C72B0")
    ax.bar(x + width / 2, tihm_pct, width, label="TIHM (wearable)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_title("Sleep Stage Label Distribution — MESA vs TIHM")
    ax.set_ylabel("Percentage of epochs (%)")
    ax.legend()
    save_figure(fig, OVERVIEW_DIR, "02_label_distribution_comparison.svg")


def plot_03_hr_comparison(data):
    feat_files = data.get("mesa_feat_files", [])
    tihm_hr    = data.get("tihm_hr")

    mesa_hr_vals = []
    for f in feat_files:
        try:
            df = pd.read_csv(f, usecols=["HR__0_Mean"])
            mesa_hr_vals.append(df["HR__0_Mean"].dropna())
        except Exception:
            pass

    fig, ax = plt.subplots()
    if mesa_hr_vals:
        mesa_hr = pd.concat(mesa_hr_vals, ignore_index=True)
        mesa_hr = mesa_hr[(mesa_hr > 20) & (mesa_hr < 200)]
        sns.kdeplot(mesa_hr, ax=ax, label="MESA HR__0_Mean (bpm)", fill=True, alpha=0.4)

    if tihm_hr is not None:
        tihm_hr_clean = tihm_hr[(tihm_hr > 20) & (tihm_hr < 200)]
        sns.kdeplot(tihm_hr_clean, ax=ax, label="TIHM heart_rate (bpm)", fill=True, alpha=0.4)

    ax.set_title("Heart Rate Distribution — MESA vs TIHM")
    ax.set_xlabel("Heart rate (bpm)")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(fig, OVERVIEW_DIR, "03_hr_distribution_comparison.svg")


def plot_04_rr_comparison(data):
    feat_files = data.get("mesa_feat_files", [])
    tihm_rr    = data.get("tihm_rr")

    mesa_rr_vals = []
    for f in feat_files:
        try:
            df = pd.read_csv(f, usecols=["Flow__0_Mean"])
            mesa_rr_vals.append(df["Flow__0_Mean"].dropna())
        except Exception:
            pass

    fig, ax = plt.subplots()
    if mesa_rr_vals:
        mesa_rr = pd.concat(mesa_rr_vals, ignore_index=True)
        sns.kdeplot(mesa_rr, ax=ax, label="MESA Flow__0_Mean (a.u.)", fill=True, alpha=0.4)

    if tihm_rr is not None:
        sns.kdeplot(tihm_rr, ax=ax, label="TIHM respiratory_rate (bpm)", fill=True, alpha=0.4)

    ax.set_title("Respiratory Signal Distribution — MESA vs TIHM")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(fig, OVERVIEW_DIR, "04_rr_distribution_comparison.svg")


def plot_05_mesa_stage_boxplot(data):
    feat_files = data.get("mesa_feat_files", [])
    if not feat_files:
        log.warning("No feature CSVs — skipping plot 05.")
        return

    stages  = ["AWAKE", "LIGHT", "DEEP", "REM"]
    records = {s: [] for s in stages}

    for f in feat_files:
        try:
            labels = pd.read_csv(f, usecols=["label"])["label"]
            total  = len(labels)
            if total == 0:
                continue
            cnt = Counter(labels)
            for s in stages:
                records[s].append(cnt.get(s, 0) / total * 100)
        except Exception:
            pass

    fig, ax = plt.subplots()
    bp_data = [records[s] for s in stages]
    ax.boxplot(bp_data, labels=stages, patch_artist=True,
               medianprops=dict(color="black", linewidth=2))
    ax.set_title("Per-Subject Sleep Stage % Distribution (MESA)")
    ax.set_ylabel("% of epochs")
    save_figure(fig, OVERVIEW_DIR, "05_mesa_stage_distribution_per_subject.svg")


def plot_06_tihm_timeline(data):
    sleep_df = data.get("tihm_sleep_df")
    pid_col  = data.get("tihm_pid_col")
    if sleep_df is None or pid_col is None:
        log.warning("TIHM sleep data not available — skipping plot 06.")
        return

    date_col = next((c for c in sleep_df.columns if "date" in c.lower()), None)
    if date_col is None:
        log.warning("No date column in TIHM sleep data — skipping plot 06.")
        return

    try:
        sleep_df[date_col] = pd.to_datetime(sleep_df[date_col], errors="coerce")
    except Exception:
        log.warning("Could not parse date column — skipping plot 06.")
        return

    ranges = (sleep_df.dropna(subset=[date_col])
              .groupby(pid_col)[date_col]
              .agg(["min", "max"])
              .sort_values("min"))

    fig, ax = plt.subplots(figsize=(12, max(4, len(ranges) * 0.4)))
    for i, (pid, row) in enumerate(ranges.iterrows()):
        ax.barh(i, (row["max"] - row["min"]).days, left=row["min"].toordinal(),
                height=0.6, color="#4C72B0", alpha=0.7)
    ax.set_yticks(range(len(ranges)))
    ax.set_yticklabels(ranges.index, fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.set_xlabel("Date")
    ax.set_title("TIHM Patient Recording Timelines")
    fig.autofmt_xdate()
    save_figure(fig, OVERVIEW_DIR, "06_tihm_patient_timeline.svg")


def plot_07_alignment_heatmap():
    # 26 aligned features × 2 datasets
    # Status: 1=Direct, 0.5=Derivable, 0=N/A
    features = [
        "Mean HR",
        "Std HR",
        "Min HR",
        "Max HR",
        "Median HR",
        "Mean SpO2",
        "Std SpO2",
        "Min SpO2",
        "Mean Resp Rate",
        "Std Resp Rate",
        "Mean Resp Amplitude",
        "Snore presence",
        "EEG Delta power",
        "EEG Theta power",
        "EEG Alpha power",
        "EEG Beta power",
        "EEG Sigma power",
        "EEG spectral entropy",
        "EOG movement",
        "EMG amplitude",
        "HR skewness",
        "HR kurtosis",
        "HR spectral entropy",
        "Flow mean",
        "Flow std",
        "Thorax amplitude",
    ]
    datasets   = ["MESA (PSG)", "TIHM (wearable)"]
    # 1=Direct, 0.5=Derivable, 0=N/A
    matrix = [
        [1,   1  ],  # Mean HR
        [1,   1  ],  # Std HR
        [1,   1  ],  # Min HR
        [1,   1  ],  # Max HR
        [1,   1  ],  # Median HR
        [1,   0  ],  # Mean SpO2
        [1,   0  ],  # Std SpO2
        [1,   0  ],  # Min SpO2
        [1,   0.5],  # Mean Resp Rate
        [1,   0.5],  # Std Resp Rate
        [1,   0  ],  # Mean Resp Amplitude
        [1,   1  ],  # Snore presence
        [1,   0  ],  # EEG Delta power
        [1,   0  ],  # EEG Theta power
        [1,   0  ],  # EEG Alpha power
        [1,   0  ],  # EEG Beta power
        [1,   0  ],  # EEG Sigma power
        [1,   0  ],  # EEG spectral entropy
        [1,   0  ],  # EOG movement
        [1,   0  ],  # EMG amplitude
        [1,   1  ],  # HR skewness
        [1,   1  ],  # HR kurtosis
        [1,   1  ],  # HR spectral entropy
        [1,   0  ],  # Flow mean
        [1,   0  ],  # Flow std
        [1,   0  ],  # Thorax amplitude
    ]

    mat = np.array(matrix)
    cmap = matplotlib.colors.ListedColormap(["#d3d3d3", "#f0ad4e", "#5cb85c"])
    bounds = [-0.25, 0.25, 0.75, 1.25]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(5, 12))
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_title("Feature Alignment: MESA vs TIHM", pad=12)

    patches = [
        mpatches.Patch(color="#5cb85c", label="Direct"),
        mpatches.Patch(color="#f0ad4e", label="Derivable"),
        mpatches.Patch(color="#d3d3d3", label="N/A"),
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    save_figure(fig, OVERVIEW_DIR, "07_feature_alignment_heatmap.svg")


def plot_08_tihm_events(data):
    labels_df = data.get("tihm_labels")
    if labels_df is None:
        log.warning("TIHM Labels.csv not available — skipping plot 08.")
        return

    type_col = next((c for c in labels_df.columns if c.lower() == "type"), None)
    if type_col is None:
        log.warning("No 'type' column in TIHM Labels.csv — skipping plot 08.")
        return

    counts = labels_df[type_col].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.4)))
    ax.barh(counts.index, counts.values, color="#4C72B0", alpha=0.8)
    ax.set_xlabel("Count")
    ax.set_title("TIHM Clinical Event Types")
    save_figure(fig, OVERVIEW_DIR, "08_tihm_clinical_events.svg")


# ---------------------------------------------------------------------------
# Section 6 — Write summary
# ---------------------------------------------------------------------------

def write_summary(lines):
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Summary saved → {SUMMARY_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    apply_thesis_style()
    import matplotlib
    import matplotlib.dates

    log.info("=" * 60)
    log.info("05_dataset_overview — MESA + TIHM")
    log.info("=" * 60)

    data  = load_data()
    lines = ["DATASET OVERVIEW — MESA + TIHM",
             f"Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}"]

    mesa_harmonized_stats(data, lines)
    mesa_feature_stats(data, lines)
    tihm_stats(data, lines)

    log.info("Generating plots ...")
    plot_01_mesa_coverage(data)
    plot_02_label_comparison(data)
    plot_03_hr_comparison(data)
    plot_04_rr_comparison(data)
    plot_05_mesa_stage_boxplot(data)
    plot_06_tihm_timeline(data)
    plot_07_alignment_heatmap()
    plot_08_tihm_events(data)

    elapsed = time.time() - t0
    lines.append(f"\n{'=' * 70}")
    lines.append(f"Total runtime: {elapsed:.1f}s")
    write_summary(lines)

    log.info(f"Done in {elapsed:.1f}s — outputs in {OVERVIEW_DIR}")


if __name__ == "__main__":
    main()
