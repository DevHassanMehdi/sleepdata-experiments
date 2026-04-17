---
name: Project Status — script completion state
description: Which scripts are done, which are next, and what the 26-feature aligned set contains
type: project
---

**Completed:**
- `scripts/02_extract_mesa_features.py` — full TSFEL extraction (14 ch × 156 features = 2185 cols + label). 350 CSVs in `outputs/features/full/`
- `scripts/05_dataset_overview.py` — 8 SVG plots, not yet run

**In progress / next:**
- `scripts/03_extract_mesa_aligned_features.py` — extract 26 harmonized features from MESA EDF (subset: HR, Flow, Snore, EKG for HRV, SpO2). Output: `outputs/features/aligned/mesa_features_aligned_XXXX.csv`
- `scripts/04_extract_tihm_features.py` — compute same 26 features per epoch from TIHM Sleep.csv. Output: `outputs/features/tihm_features.csv`

**Stubs (not started):** scripts 06 (ML training), 07 (harmonization), 08 (comparison)

**Ignored:** `outputs/features/wearable/` — legacy, do not reference or delete

**Why:** The 26-feature aligned set is the core thesis contribution enabling cross-dataset comparison.

**How to apply:** Script 03 and 04 are the immediate priority. Do not re-create deleted files: utils/data_loader.py, utils/metrics.py.
