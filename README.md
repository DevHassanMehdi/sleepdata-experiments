# Sleep Stage Classification Across Clinical PSG and Wearable IoT Data

Master's thesis research repository comparing ML sleep staging performance between MESA (clinical PSG) and TIHM (wearable IoT).

## Datasets

| Name | Type         | Source                                                    |
|------|--------------|-----------------------------------------------------------|
| MESA | Clinical PSG | NSRR — Multi-Ethnic Study of Atherosclerosis              |
| TIHM | Wearable IoT | Zenodo — Technology Integrated Health Management for Dementia |

## Project Structure

```
sleepdata-experiments/
├── environment.yml          # Full conda environment spec
├── environment_minimal.yml  # Minimal top-level deps only
├── requirements.txt         # pip dependencies
├── data/
│   ├── mesa/
│   │   ├── edf/             # MESA EDF recordings (not committed)
│   │   ├── annotations/     # NSRR XML annotation files (not committed)
│   │   └── datasets/        # MESA harmonized CSV (not committed)
│   └── tihm/                # TIHM CSV files (not committed)
├── outputs/
│   ├── features/
│   │   └── full/            # mesa_features_full_XXXX.csv (one per subject)
│   └── overview/            # SVG plots + summary text
├── scripts/                 # Pipeline scripts (run in order)
├── utils/
│   ├── logger.py
│   └── plotting.py
└── logs/
```

## Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate sleepdata

# Install NSRR download tool (requires Ruby, included in conda env)
gem install nsrr
```

### NSRR token

Create `~/.nsrr_token` (or a `.env` file with `NSRR_TOKEN=<your-token>`) before running `01_download_dataset.py`.

## Pipeline

| Script                              | Description                                    | Status      |
|-------------------------------------|------------------------------------------------|-------------|
| `01_download_dataset.py`            | Download MESA EDFs + TIHM CSV files            | Complete    |
| `02_extract_mesa_features.py`       | Extract TSFEL features from MESA EDF (full PSG)| Complete    |
| `03_extract_mesa_aligned_features.py` | Extract 26 aligned features from MESA EDF   | Placeholder |
| `04_extract_tihm_features.py`       | Extract features from TIHM CSV files           | Placeholder |
| `05_dataset_overview.py`            | Dataset profiling, statistics, visualisations  | Complete    |
| `06_model_training.py`              | Train ML classifiers                           | Planned     |
| `07_harmonization.py`               | Harmonize features across datasets             | Planned     |
| `08_comparison.py`                  | Cross-dataset performance comparison           | Planned     |

### Run feature extraction (batch)

```bash
conda activate sleepdata
python scripts/02_extract_mesa_features.py --subjects 350
```

### Run dataset overview

```bash
python scripts/05_dataset_overview.py
# Outputs: outputs/overview/  (8 SVGs + dataset_overview_summary.txt)
```

## Current Status

- MESA feature extraction complete (~350 subjects, full 14-channel PSG, 2,185 features/epoch).
- Dataset overview script covers MESA harmonized stats, extracted feature stats, TIHM stats, and 8 visualisation plots.
- Next: define and extract 26 aligned features for cross-dataset comparison (`03`, `04`).
