# Evaluating Machine Learning Performance Across Clinical PSG and Wearable Sleep Data: Feature Reduction and Harmonization Analysis

Master's thesis research repository.

## Overview

This project systematically compares machine learning model performance across four clinical polysomnography (PSG) datasets and one wearable IoT dataset, then evaluates whether data harmonization improves cross-device generalization.

## Datasets

| Name  | Type          | Source                                      |
|-------|---------------|---------------------------------------------|
| SHHS  | Clinical PSG  | NSRR — Sleep Heart Health Study             |
| MrOS  | Clinical PSG  | NSRR — Osteoporotic Fractures in Men Study  |
| MESA  | Clinical PSG  | NSRR — Multi-Ethnic Study of Atherosclerosis|
| SSC   | Clinical PSG  | Stanford Sleep Cohort                       |
| TIHM  | Wearable IoT  | Zenodo — Technology Integrated Health Management for Dementia |

## Setup

```bash
pip install -r requirements.txt
```

## Downloading Data

```bash
# TIHM (Zenodo — no account required)
python downloaders/download_tihm.py

# NSRR datasets (requires NSRR account + token)
python downloaders/download_nsrr.py
```

See [downloaders/README.md](downloaders/README.md) for details.

## Pipeline Stages

Run scripts in order. Each script reads from `data/<dataset>/` and writes to `outputs/<dataset>/`.

| Stage | Script                          | Description                                      |
|-------|---------------------------------|--------------------------------------------------|
| 01    | `scripts/01_dataset_profiling.py`   | Load CSVs, log shapes, dtypes, missing values    |
| 02    | `scripts/02_feature_inventory.py`   | Categorize all features by keyword matching      |
| 03    | `scripts/03_data_quality.py`        | Missing %, outlier counts, quality report        |
| 04    | `scripts/04_eda.py`                 | Histograms, correlation, PCA / t-SNE / UMAP      |
| 05    | `scripts/05_feature_alignment.py`   | Cross-dataset feature alignment matrix           |
| 06    | `scripts/06_model_training.py`      | ML model training per dataset *(TBD)*            |
| 07    | `scripts/07_harmonization.py`       | Data harmonization across datasets *(TBD)*       |
| 08    | `scripts/08_comparison.py`          | Cross-dataset performance comparison *(TBD)*     |

Change `DATASET_NAME` at the top of any script to switch between datasets.

## Project Structure

```
sleepdata-experiments/
├── config.py              # Central configuration
├── requirements.txt
├── data/                  # Raw dataset files (not committed)
│   ├── tihm/
│   ├── shhs/
│   ├── mros/
│   ├── mesa/
│   └── ssc/
├── outputs/               # Generated figures and tables (not committed)
├── logs/                  # Runtime logs (not committed)
├── utils/                 # Shared utilities
├── scripts/               # Pipeline stage scripts
└── downloaders/           # Dataset download helpers
```
