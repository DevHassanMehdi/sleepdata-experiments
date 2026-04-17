---
name: Project Overview — sleepdata-experiments
description: Master's thesis: cross-dataset sleep-stage ML between MESA (PSG) and TIHM (wearable)
type: project
---

**Goal:** Train ML models on one dataset, evaluate on the other after harmonizing to 26 aligned features.

**Datasets:**
- MESA — clinical full-night PSG, ~350 subjects, EDF files + NSRR XML annotations, 14 channels
- TIHM — wearable IoT, dementia patients, Zenodo CSVs (Sleep.csv, Demographics.csv, Activity.csv, Labels.csv, Physiology.csv)

**Label scheme (4-class):** AWAKE, LIGHT, DEEP, REM

**Epoch granularity:** 1-minute epochs (MESA: two 30s PSG windows paired; TIHM: native)

**Feature column naming:** `{CHANNEL}__0_{FeatureName}` — double underscore + `0_` is TSFEL's internal indexing

**Key channels for alignment:** HR, Flow (respiratory proxy), Snore, EKG (HRV), SpO2

**UNC paths:**
- Bash: `//wsl.localhost/Ubuntu/home/devhassan/projects/sleepdata-experiments/`
- Read/Write/Edit tools: `\\wsl.localhost\Ubuntu\home\devhassan\projects\sleepdata-experiments\`
- Never use `/home/devhassan/...` in Bash — resolves to empty Windows home dir

**Why:** Cross-dataset generalization is the thesis contribution; harmonized 26-feature set bridges PSG and wearable modalities.

**How to apply:** Always use UNC paths in Bash. Always maintain the 26-feature aligned set as the canonical bridge between datasets.
