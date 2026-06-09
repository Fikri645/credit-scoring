"""Deploy the Gradio demo to HuggingFace Spaces.

Prereqs:
    pip install huggingface_hub
    set HF_TOKEN=hf_xxx           # a token with *write* access

Run:
    python deploy/deploy_hf_space.py            # default repo id below
    python deploy/deploy_hf_space.py fikri0o0/credit-scoring

Uploads only the lean files the demo needs (model + schema + demo applicants +
app + minimal src), renames the lean requirements / HF README into place, and
creates the Space as a Gradio SDK app on free CPU hardware.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "fikri0o0/credit-scoring"

# (local path, path inside the Space repo)
UPLOADS = [
    ("app.py", "app.py"),
    ("src/__init__.py", "src/__init__.py"),
    ("src/config.py", "src/config.py"),
    ("models/model_lgbm.txt", "models/model_lgbm.txt"),
    ("models/feature_schema.joblib", "models/feature_schema.joblib"),
    ("models/analysis_threshold.json", "models/analysis_threshold.json"),
    ("artifacts/demo_applicants.parquet", "artifacts/demo_applicants.parquet"),
    ("deploy/requirements-spaces.txt", "requirements.txt"),
    ("deploy/README_hf.md", "README.md"),
]


def main() -> int:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN=hf_... (a write token) first.", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    print(f"[hf] creating Space {repo_id} (gradio, cpu-basic) …")
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                    exist_ok=True)

    for local, remote in UPLOADS:
        p = ROOT / local
        if not p.exists():
            print(f"[hf] MISSING {local} — did you run train + analysis?",
                  file=sys.stderr)
            return 1
        print(f"[hf] upload {local} -> {remote}")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=remote,
                        repo_id=repo_id, repo_type="space")

    print(f"[hf] done -> https://huggingface.co/spaces/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
