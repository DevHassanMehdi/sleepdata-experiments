# =============================================================================
# utils/download_dataset.py
# -----------------------------------------------------------------------------
# Unified dataset downloader.
#
# Usage:
#   python3 utils/download_dataset.py --datasets tihm
#   python3 utils/download_dataset.py --datasets mesa mros shhs
#   python3 utils/download_dataset.py --datasets tihm mesa mros shhs
#
# TIHM  -- downloaded from Zenodo (no account required)
# NSRR  -- downloaded via the nsrr Ruby gem (mesa, mros, shhs)
#          Requires: gem install nsrr
#
# WARNING: The NSRR token is personal. Do NOT commit to a public repo.
#          Move it to a .env file and load with python-dotenv for safety.
# =============================================================================

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# NSRR configuration
# ---------------------------------------------------------------------------
NSRR_TOKEN    = "30343-g8xnH2hn44e6bnxwLqKn"
NSRR_DATASETS = set(config.NSRR_DATASETS)

# ---------------------------------------------------------------------------
# TIHM configuration
# ---------------------------------------------------------------------------
TIHM_URL  = "https://zenodo.org/api/records/7622128/files/TIHM_Dataset.zip/content"
TIHM_ZIP  = ROOT_DIR / "TIHM_Dataset.zip"
TIHM_CSVS = ["Activity.csv", "Demographics.csv", "Labels.csv", "Physiology.csv", "Sleep.csv"]
CHUNK_SIZE = 1024 * 1024  # 1 MB


# =============================================================================
# TIHM
# =============================================================================

def download_tihm(logger):
    target_dir = config.DATASET_PATHS["tihm"]

    logger.info("Downloading TIHM from Zenodo...")
    response = requests.get(TIHM_URL, stream=True, timeout=120)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    logger.info("File size: %.1f MB", total_bytes / 1024 / 1024)

    with open(TIHM_ZIP, "wb") as f, tqdm(
        desc="  TIHM_Dataset.zip",
        total=total_bytes,
        unit="B", unit_scale=True, unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))

    logger.info("Extracting...")
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(TIHM_ZIP, "r") as zf:
        for csv_name in TIHM_CSVS:
            matches = [n for n in zf.namelist() if n.endswith(csv_name)]
            if not matches:
                logger.warning("Not found in zip: %s", csv_name)
                continue
            dest = target_dir / csv_name
            with zf.open(matches[0]) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("Extracted -> %s", dest)

    TIHM_ZIP.unlink()
    logger.info("TIHM ready in: %s", target_dir)
    return True


# =============================================================================
# NSRR
# =============================================================================

def check_nsrr_gem(logger):
    try:
        result = subprocess.run(["nsrr", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("nsrr gem: %s", result.stdout.strip())
            return True
    except FileNotFoundError:
        pass
    print("\nError: nsrr gem not found. Install with:  gem install nsrr\n")
    return False


def download_nsrr_dataset(dataset_name, logger):
    data_dir = config.DATASET_PATHS[dataset_name]
    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["nsrr", "download", f"{dataset_name}/datasets", f"--token={NSRR_TOKEN}"]
    logger.info("Running: %s", " ".join(cmd))

    process = subprocess.Popen(
        cmd, cwd=str(data_dir.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in process.stdout:
        line = line.rstrip()
        print(line)
        logger.debug(line)
    process.wait()

    if process.returncode == 0:
        logger.info("SUCCESS: %s", dataset_name)
        return True
    else:
        logger.error("FAILED: %s (exit code %d)", dataset_name, process.returncode)
        return False


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="python3 utils/download_dataset.py",
        description="Download one or more sleep datasets.",
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True, metavar="NAME",
        help="Dataset names: tihm, mesa, mros, shhs, ssc",
    )
    args = parser.parse_args()

    # Strip accidental commas (e.g. --datasets tihm, mesa)
    requested = [d.strip().rstrip(",").lower() for d in args.datasets]

    valid = set(config.DATASET_PATHS.keys())
    unknown = [d for d in requested if d not in valid]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Valid: {sorted(valid)}")
        sys.exit(1)

    logger = get_logger("download_dataset", "_".join(requested))

    results = {}
    nsrr_checked = False

    for dataset in requested:
        sep = "=" * 50
        print(f"\n{sep}\n  {dataset.upper()}\n{sep}")

        if dataset == "tihm":
            try:
                results[dataset] = download_tihm(logger)
            except Exception as exc:
                logger.error("Error: %s", exc, exc_info=True)
                results[dataset] = False

        elif dataset in NSRR_DATASETS:
            if not nsrr_checked:
                if not check_nsrr_gem(logger):
                    for d in requested:
                        if d in NSRR_DATASETS and d not in results:
                            results[d] = False
                    break
                nsrr_checked = True
            results[dataset] = download_nsrr_dataset(dataset, logger)

        else:
            logger.warning("No downloader implemented for %r yet.", dataset)
            results[dataset] = False

    sep = "=" * 50
    print(f"\n{sep}\n  Summary\n{sep}")
    for d, ok in results.items():
        print(f"  {d:<10} {'OK' if ok else 'FAILED'}")
    print(sep)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
