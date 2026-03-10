# =============================================================================
# utils/download.py
# -----------------------------------------------------------------------------
# Unified dataset downloader.  Pass one or more dataset names to download.
#
# Usage:
#   python3 utils/download.py --datasets tihm
#   python3 utils/download.py --datasets mesa mros shhs
#   python3 utils/download.py --datasets tihm mesa mros shhs ssc
#
# TIHM  — downloaded from Zenodo (no account required)
# NSRR  — downloaded via the nsrr Ruby gem (mesa, mros, shhs)
#          Requires:  gem install nsrr
#
# WARNING: The NSRR token below is personal. Do NOT commit it to a public repo.
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

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# NSRR configuration — move token to .env in future
# ---------------------------------------------------------------------------
NSRR_TOKEN   = "30343-g8xnH2hn44e6bnxwLqKn"
NSRR_DATASETS = set(config.NSRR_DATASETS)   # {"mesa", "mros", "shhs"}

# ---------------------------------------------------------------------------
# TIHM configuration
# ---------------------------------------------------------------------------
TIHM_URL = "https://zenodo.org/api/records/7622128/files/TIHM_Dataset.zip/content"
TIHM_ZIP = ROOT_DIR / "TIHM_Dataset.zip"
TIHM_CSVS = [
    "Activity.csv",
    "Demographics.csv",
    "Labels.csv",
    "Physiology.csv",
    "Sleep.csv",
]
CHUNK_SIZE = 1024 * 1024  # 1 MB


# =============================================================================
# TIHM downloader
# =============================================================================

def download_tihm(logger):
    """Download and extract the TIHM dataset from Zenodo."""
    target_dir = config.DATASET_PATHS["tihm"]

    # --- Download ---
    logger.info("Downloading TIHM from Zenodo: %s", TIHM_URL)
    response = requests.get(TIHM_URL, stream=True, timeout=120)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    logger.info("File size: %.1f MB", total_bytes / 1024 / 1024)

    with open(TIHM_ZIP, "wb") as f, tqdm(
        desc="  Downloading TIHM_Dataset.zip",
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))

    logger.info("Download complete → %s", TIHM_ZIP)

    # --- Extract ---
    logger.info("Extracting zip...")
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(TIHM_ZIP, "r") as zf:
        all_names = zf.namelist()
        for csv_name in TIHM_CSVS:
            matches = [n for n in all_names if n.endswith(csv_name)]
            if not matches:
                logger.warning("Not found in zip: %s", csv_name)
                continue
            dest = target_dir / csv_name
            with zf.open(matches[0]) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("Extracted → %s", dest)

    # --- Cleanup ---
    TIHM_ZIP.unlink()
    logger.info("Removed zip archive.")
    logger.info("TIHM ready in: %s", target_dir)
    return True


# =============================================================================
# NSRR downloader
# =============================================================================

def check_nsrr_gem(logger):
    """Return True if the nsrr Ruby gem is available, else print instructions."""
    try:
        result = subprocess.run(
            ["nsrr", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            logger.info("nsrr gem found: %s", result.stdout.strip())
            return True
    except FileNotFoundError:
        pass

    print(
        "\nError: nsrr gem not found.\n"
        "Install with:  gem install nsrr\n"
    )
    return False


def download_nsrr_dataset(dataset_name, logger):
    """Run nsrr download for one NSRR dataset, streaming output live."""
    data_dir = config.DATASET_PATHS[dataset_name]
    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["nsrr", "download", f"{dataset_name}/datasets", f"--token={NSRR_TOKEN}"]
    logger.info("Running: %s  (cwd: %s)", " ".join(cmd), data_dir)

    try:
        process = subprocess.Popen(
            cmd, cwd=str(data_dir),
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

    except Exception as exc:
        logger.error("Error downloading %s: %s", dataset_name, exc, exc_info=True)
        return False


# =============================================================================
# CLI entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="python3 utils/download.py",
        description="Download one or more sleep datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available datasets:\n"
            "  tihm          Wearable IoT — Zenodo (no account needed)\n"
            "  mesa          NSRR — Multi-Ethnic Study of Atherosclerosis\n"
            "  mros          NSRR — Osteoporotic Fractures in Men\n"
            "  shhs          NSRR — Sleep Heart Health Study\n\n"
            "Examples:\n"
            "  python3 utils/download.py --datasets tihm\n"
            "  python3 utils/download.py --datasets mesa mros shhs\n"
            "  python3 utils/download.py --datasets tihm mesa mros shhs\n"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="NAME",
        help="One or more dataset names to download.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Strip any accidental commas (e.g. --datasets tihm, mesa)
    requested = [d.strip().rstrip(",").lower() for d in args.datasets]

    # Validate all names up front
    valid = set(config.DATASET_PATHS.keys())
    unknown = [d for d in requested if d not in valid]
    if unknown:
        print(f"Error: unknown dataset(s): {', '.join(unknown)}")
        print(f"Valid options: {', '.join(sorted(valid))}")
        sys.exit(1)

    logger = get_logger("download", "_".join(requested))
    logger.info("Datasets to download: %s", requested)

    results = {}

    for dataset in requested:
        print(f"\n{'='*55}")
        print(f"  Downloading: {dataset.upper()}")
        print(f"{'='*55}")

        if dataset == "tihm":
            try:
                results[dataset] = download_tihm(logger)
            except requests.HTTPError as exc:
                logger.error("HTTP error: %s", exc)
                results[dataset] = False
            except Exception as exc:
                logger.error("Unexpected error: %s", exc, exc_info=True)
                results[dataset] = False

        elif dataset in NSRR_DATASETS:
            if not check_nsrr_gem(logger):
                results[dataset] = False
                continue
            results[dataset] = download_nsrr_dataset(dataset, logger)

        else:
            # ssc or future datasets without a downloader yet
            logger.warning("No downloader implemented for '%s' yet.", dataset)
            results[dataset] = False

    # Summary
    print(f"\n{'='*55}")
    print("  Download Summary")
    print(f"{'='*55}")
    for dataset, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {dataset:<10} {status}")
    print(f"{'='*55}\n")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
