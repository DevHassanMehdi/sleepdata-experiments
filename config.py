# =============================================================================
# config.py — Central configuration for the sleepdata-experiments project
# =============================================================================
from pathlib import Path

# Root of the repository (directory containing this file)
ROOT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_PATHS = {
    "tihm":  ROOT_DIR / "data" / "tihm",
    "shhs":  ROOT_DIR / "data" / "shhs",
    "mros":  ROOT_DIR / "data" / "mros",
    "mesa":  ROOT_DIR / "data" / "mesa",
    "ssc":   ROOT_DIR / "data" / "ssc",
}

# ---------------------------------------------------------------------------
# Output and logging directories
# ---------------------------------------------------------------------------
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR    = ROOT_DIR / "logs"

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
PLOT_FORMAT  = "svg"
RANDOM_SEED  = 42

# Datasets available through the NSRR (National Sleep Research Resource)
NSRR_DATASETS = ["mesa", "mros", "shhs"]


# ---------------------------------------------------------------------------
# Quick sanity check — run `python config.py` to verify paths
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ROOT_DIR   :", ROOT_DIR)
    print("OUTPUTS_DIR:", OUTPUTS_DIR)
    print("LOGS_DIR   :", LOGS_DIR)
    print("\nDataset paths:")
    for name, path in DATASET_PATHS.items():
        status = "exists" if path.exists() else "missing"
        print(f"  {name:<6} -> {path}  [{status}]")
