"""Download the Home Credit 2024 competition data via **kagglehub**.

Kaggle migrated competition endpoints to an API that the legacy ``kaggle``
1.6.x client can no longer authenticate against, and the 2.x CLI needs an
interactive OAuth flow. The current, non-interactive path is an **access
token** (``KGAT_...``) read by ``kagglehub`` (>= 0.4.1).

Auth (either works; kagglehub checks both):
  * env var  ``KAGGLE_API_TOKEN=KGAT_...``
  * file     ``~/.kaggle/access_token`` containing the token

To keep the ~27 GB off the small C: drive, the kagglehub cache is forced onto
the project drive (``data/kagglehub_cache``) and the parquet files are then
moved next to where the pipeline expects them (``data/raw/parquet_files``).

Usage:  ``python -m src.download_data``
"""
from __future__ import annotations

import os
import shutil
import sys

from src import config

# Force kagglehub's cache onto the project drive *before* importing kagglehub.
os.environ.setdefault("KAGGLEHUB_CACHE", str(config.DATA_DIR / "kagglehub_cache"))


def main() -> int:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if config.TRAIN_DIR.exists() and any(config.TRAIN_DIR.glob("*.parquet")):
        print(f"[download] Train parquet already present in {config.TRAIN_DIR} — skipping.")
        return 0

    try:
        import kagglehub
    except ImportError:
        print("[download] kagglehub not installed: pip install 'kagglehub>=0.4.1'",
              file=sys.stderr)
        return 1

    if not (os.environ.get("KAGGLE_API_TOKEN")
            or (config.DATA_DIR.home() / ".kaggle" / "access_token").exists()):
        print("[download] No access token found. Set KAGGLE_API_TOKEN=KGAT_... or "
              "save it to ~/.kaggle/access_token", file=sys.stderr)
        return 1

    print(f"[download] Fetching '{config.COMPETITION}' via kagglehub "
          f"(cache: {os.environ['KAGGLEHUB_CACHE']}) …")
    try:
        comp_path = kagglehub.competition_download(config.COMPETITION)
    except Exception as exc:  # noqa: BLE001
        print(f"[download] kagglehub download failed: {exc}", file=sys.stderr)
        return 1

    # Move the downloaded sub-folders next to where the pipeline expects them.
    # Same-drive moves are instantaneous renames (no 27 GB copy).
    src_root = config.RAW_DIR.__class__(comp_path)
    for child in src_root.iterdir():
        dest = config.RAW_DIR / child.name
        if dest.exists():
            continue
        print(f"[download] moving {child.name} -> {dest}")
        shutil.move(str(child), str(dest))

    # Drop the now-empty kagglehub cache tree for this competition.
    try:
        shutil.rmtree(config.DATA_DIR / "kagglehub_cache" / "competitions",
                      ignore_errors=True)
    except OSError:
        pass

    if config.TRAIN_DIR.exists() and any(config.TRAIN_DIR.glob("*.parquet")):
        n = len(list(config.TRAIN_DIR.glob("*.parquet")))
        print(f"[download] Done — {n} train parquet shards in {config.TRAIN_DIR}.")
        return 0
    print(f"[download] WARNING: expected parquet not found under {config.TRAIN_DIR}. "
          f"Inspect {config.RAW_DIR}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
