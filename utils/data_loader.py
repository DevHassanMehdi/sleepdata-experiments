# =============================================================================
# utils/data_loader.py — CSV loading utilities
# =============================================================================
import sys
from pathlib import Path

import pandas as pd

# Add project root to path so config can be imported from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# load_csvs
# ---------------------------------------------------------------------------

def load_csvs(folder_path):
    """Recursively load all CSV files found under *folder_path*.

    Parameters
    ----------
    folder_path : str or Path
        Directory to search (searched recursively for ``**/*.csv``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ``filename`` (stem only, e.g. ``"Activity"``) to its
        corresponding DataFrame.  Files that cannot be parsed are skipped with
        a warning printed to stdout.

    Raises
    ------
    FileNotFoundError
        If *folder_path* does not exist.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Data folder not found: {folder_path}")

    csv_files = sorted(folder_path.rglob("*.csv"))

    if not csv_files:
        print(f"[WARNING] No CSV files found in: {folder_path}")
        return {}

    dataframes = {}
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            dataframes[csv_path.stem] = df
        except Exception as exc:
            print(f"[WARNING] Could not read {csv_path.name}: {exc}")

    print(f"[INFO] Loaded {len(dataframes)} CSV file(s) from {folder_path}")
    return dataframes


# ---------------------------------------------------------------------------
# get_dataset_path
# ---------------------------------------------------------------------------

def get_dataset_path(dataset_name):
    """Return the data directory Path for a named dataset.

    Parameters
    ----------
    dataset_name : str
        One of the keys in ``config.DATASET_PATHS``
        (``"tihm"``, ``"shhs"``, ``"mros"``, ``"mesa"``, ``"ssc"``).

    Returns
    -------
    Path

    Raises
    ------
    ValueError
        If *dataset_name* is not a recognised dataset key.
    """
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in config.DATASET_PATHS:
        valid = ", ".join(sorted(config.DATASET_PATHS.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options are: {valid}"
        )
    return config.DATASET_PATHS[dataset_name]
