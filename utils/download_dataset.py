# =============================================================================
# utils/download_dataset.py
# =============================================================================
# Download MESA EDF + annotation files and/or TIHM CSV files.
#
# Usage:
#   python utils/download_dataset.py --dataset all              # default
#   python utils/download_dataset.py --dataset mesa             # 5 subjects (default)
#   python utils/download_dataset.py --dataset mesa --subjects 20
#   python utils/download_dataset.py --dataset tihm
#
# NSRR token is read from .env (NSRR_TOKEN=...).
# =============================================================================

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TIHM_DIR  = ROOT_DIR / "data/tihm"
EDF_DIR   = ROOT_DIR / "data/mesa/edf"
ANNOT_DIR = ROOT_DIR / "data/mesa/annotations"
LOGS_DIR  = ROOT_DIR / "logs"

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
MESA_EDF_BASE   = "https://sleepdata.org/datasets/mesa/files/m/browser/polysomnography/edfs"
MESA_ANNOT_BASE = "https://sleepdata.org/datasets/mesa/files/m/browser/polysomnography/annotations-events-nsrr"
TIHM_ZENODO_URL = "https://zenodo.org/api/records/7622128/files/TIHM_Dataset.zip/content"
TIHM_CSV_FILES  = ["Sleep.csv", "Demographics.csv", "Physiology.csv", "Activity.csv", "Labels.csv"]

CHUNK_SIZE    = 1024 * 1024   # 1 MB
EDF_MIN_BYTES = 50 * 1024 * 1024  # 50 MB — anything smaller is corrupt / not found


# =============================================================================
# Token
# =============================================================================

def load_token() -> str:
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("NSRR_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                if token and token != "your_token_here":
                    return token
    print("[ERROR] NSRR_TOKEN not found in .env")
    print("        Create .env with:  NSRR_TOKEN=your_token_here")
    sys.exit(1)


# =============================================================================
# Logger
# =============================================================================

def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("download_dataset")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(LOGS_DIR / "download_dataset.txt", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)
    return logger


# =============================================================================
# Stream download helper
# =============================================================================

def stream_download(url: str, dest: Path, token: str, desc: str) -> bool:
    """
    Stream-download url → dest with a tqdm progress bar.
    Appends auth_token as a URL parameter.
    Returns True on success, False on any error.
    """
    params = {"auth_token": token} if token else {}
    try:
        resp = requests.get(url, params=params, stream=True,
                            timeout=300, allow_redirects=True)
        if resp.status_code != 200:
            return False

        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as fh, tqdm(
            desc=f"  {desc}",
            total=total or None,
            unit="B", unit_scale=True, unit_divisor=1024,
            leave=False,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                fh.write(chunk)
                bar.update(len(chunk))
        return True

    except Exception:
        if dest.exists():
            dest.unlink()
        return False


# =============================================================================
# MESA
# =============================================================================

MAX_SUBJECT_ID = 9999   # upper bound to prevent infinite loops


def download_mesa(n_subjects: int, token: str, logger: logging.Logger) -> dict:
    counts = {"ok": 0, "skipped": 0, "not_found": 0}
    EDF_DIR.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    found = 0   # subjects successfully downloaded or already present
    i = 0
    while found < n_subjects:
        i += 1
        if i > MAX_SUBJECT_ID:
            print(f"  [STOP] Reached max subject ID {MAX_SUBJECT_ID} with only {found}/{n_subjects} found.")
            logger.warning("Reached MAX_SUBJECT_ID=%d with found=%d/%d", MAX_SUBJECT_ID, found, n_subjects)
            break
        sid      = f"{i:04d}"
        edf_name = f"mesa-sleep-{sid}.edf"
        xml_name = f"mesa-sleep-{sid}-nsrr.xml"
        edf_dest = EDF_DIR   / edf_name
        xml_dest = ANNOT_DIR / xml_name

        # --- Skip if EDF already downloaded and large enough ---
        if edf_dest.exists() and edf_dest.stat().st_size >= EDF_MIN_BYTES:
            mb = edf_dest.stat().st_size / 1_048_576
            print(f"  [SKIP] subject {sid} already downloaded  ({mb:.1f} MB)")
            logger.info("Skipped subject %s — EDF already present (%.1f MB)", sid, mb)
            counts["skipped"] += 1
            found += 1
            continue

        # Remove any existing small/corrupt EDF before re-downloading
        if edf_dest.exists():
            edf_dest.unlink()

        # --- Download EDF ---
        edf_ok = stream_download(
            f"{MESA_EDF_BASE}/{edf_name}", edf_dest, token, edf_name
        )

        if not edf_ok or not edf_dest.exists() or edf_dest.stat().st_size < EDF_MIN_BYTES:
            if edf_dest.exists():
                edf_dest.unlink()
            # Only treat first subject failure as a possible auth issue
            if i == 1 and counts["ok"] == 0:
                print(f"  [SKIP] subject {sid} not found or missing "
                      f"— if this continues, check NSRR_TOKEN in .env")
            else:
                print(f"  [SKIP] subject {sid} not found or missing")
            logger.warning("Subject %s: EDF not found or too small", sid)
            counts["not_found"] += 1
            continue

        found += 1
        edf_mb = edf_dest.stat().st_size / 1_048_576
        print(f"  ✓  {edf_name}  ({edf_mb:.1f} MB)")
        logger.info("Downloaded %s  (%.1f MB)", edf_name, edf_mb)

        # --- Download XML annotation ---
        xml_ok = stream_download(
            f"{MESA_ANNOT_BASE}/{xml_name}", xml_dest, token, xml_name
        )

        if xml_ok and xml_dest.exists():
            xml_kb = xml_dest.stat().st_size / 1_024
            print(f"  ✓  {xml_name}  ({xml_kb:.1f} KB)")
            logger.info("Downloaded %s  (%.1f KB)", xml_name, xml_kb)
        else:
            print(f"  [WARN] {xml_name} not downloaded")
            logger.warning("Subject %s: XML annotation not downloaded", sid)

        counts["ok"] += 1

    return counts


# =============================================================================
# TIHM
# =============================================================================

def download_tihm(logger: logging.Logger) -> dict:
    TIHM_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skipped": 0, "not_found": 0}

    already = [f for f in TIHM_CSV_FILES if (TIHM_DIR / f).exists()]
    missing = [f for f in TIHM_CSV_FILES if not (TIHM_DIR / f).exists()]

    if already:
        print(f"  [SKIP] already present: {', '.join(already)}")
        counts["skipped"] += len(already)

    if not missing:
        print("  All TIHM files already downloaded.")
        return counts

    print(f"  Downloading TIHM zip from Zenodo ({len(missing)} files needed) ...")
    logger.info("TIHM: need %s", missing)

    zip_path = ROOT_DIR / "TIHM_Dataset.zip"
    ok = stream_download(TIHM_ZENODO_URL, zip_path, token="", desc="TIHM_Dataset.zip")
    if not ok:
        print("  [FAIL] Could not download TIHM zip from Zenodo.")
        counts["not_found"] += len(missing)
        return counts

    print("  Extracting CSV files ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            for csv_name in missing:
                matches = [n for n in names if n.endswith(csv_name)]
                if not matches:
                    print(f"  [WARN] {csv_name} not found in zip")
                    counts["not_found"] += 1
                    continue
                dest = TIHM_DIR / csv_name
                with zf.open(matches[0]) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                kb = dest.stat().st_size / 1_024
                print(f"  ✓  {csv_name}  ({kb:.0f} KB)")
                logger.info("Extracted %s (%.0f KB)", csv_name, kb)
                counts["ok"] += 1
    except zipfile.BadZipFile as exc:
        print(f"  [FAIL] Bad zip: {exc}")
        counts["not_found"] += len(missing)
    finally:
        zip_path.unlink(missing_ok=True)

    return counts


# =============================================================================
# Main
# =============================================================================

def print_summary(results: dict):
    sep = "=" * 50
    print(f"\n{sep}\n  DOWNLOAD SUMMARY\n{sep}")
    for ds, r in results.items():
        print(f"  {ds.upper():<8}  "
              f"downloaded={r['ok']}  skipped={r['skipped']}  not_found={r['not_found']}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        prog="python utils/download_dataset.py",
        description="Download MESA EDF + annotations and/or TIHM CSV files.",
    )
    parser.add_argument(
        "--dataset", choices=["mesa", "tihm", "all"], default="all",
        help="Which dataset to download (default: all).",
    )
    parser.add_argument(
        "--subjects", type=int, default=5, metavar="N",
        help="Number of MESA subjects to download (default: 5).",
    )
    args = parser.parse_args()

    if args.subjects < 1:
        parser.error("--subjects must be >= 1")

    logger = setup_logger()
    token  = load_token()
    results = {}

    run_mesa = args.dataset in ("mesa", "all")
    run_tihm = args.dataset in ("tihm", "all")

    logger.info("dataset=%s  subjects=%d", args.dataset, args.subjects)

    if run_mesa:
        sep = "=" * 50
        print(f"\n{sep}\n  MESA — {args.subjects} subject(s)\n{sep}")
        logger.info("MESA: downloading %d subjects", args.subjects)
        results["mesa"] = download_mesa(args.subjects, token, logger)

    if run_tihm:
        sep = "=" * 50
        print(f"\n{sep}\n  TIHM — Zenodo\n{sep}")
        logger.info("TIHM: starting download")
        results["tihm"] = download_tihm(logger)

    print_summary(results)
    logger.info("Done: %s", {ds: dict(r) for ds, r in results.items()})


if __name__ == "__main__":
    main()
