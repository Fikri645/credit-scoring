"""Download and unzip the Home Credit 2024 competition data via the Kaggle API.

Usage (from repo root, venv active):

    python -m src.download_data

Requires ``~/.kaggle/kaggle.json`` and that you have accepted the competition
rules at:
    https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/rules

The download is ~7 GB zipped (~26 GB unzipped). Parquet base tables land in
``data/raw/parquet_files/{train,test}/``.
"""
from __future__ import annotations

import sys
import zipfile

from src import config


def main() -> int:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.DATA_DIR / f"{config.COMPETITION}.zip"

    if config.TRAIN_DIR.exists() and any(config.TRAIN_DIR.glob("*.parquet")):
        print(f"[download] Train parquet already present in {config.TRAIN_DIR} — skipping.")
        return 0

    if not zip_path.exists():
        print(f"[download] Fetching competition '{config.COMPETITION}' via Kaggle API…")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            # quiet=False streams a progress bar; downloads the zipped bundle.
            api.competition_download_files(
                config.COMPETITION, path=str(config.DATA_DIR), quiet=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[download] Kaggle download failed: {exc}\n"
                "Make sure you have:\n"
                "  1. ~/.kaggle/kaggle.json with a valid token\n"
                "  2. Accepted the competition rules on the website:\n"
                f"     https://www.kaggle.com/competitions/{config.COMPETITION}/rules\n",
                file=sys.stderr,
            )
            return 1

    print(f"[download] Unzipping {zip_path.name} → {config.RAW_DIR} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.RAW_DIR)
    print("[download] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
