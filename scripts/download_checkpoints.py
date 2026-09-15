#!/usr/bin/env python3
"""
Model checkpoint downloader for music-gpu-lab.
Downloads pinned ACE-Step 1.5 XL-SFT (4B DiT) and 5Hz-LM-4B checkpoints from HuggingFace.
Features cache hit detection, download size estimation, and file integrity validation.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Exact pinned Hugging Face repositories and revision commit SHAs
PINNED_CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "acestep-v15-xl-sft": {
        "repo_id": "ACE-Step/acestep-v15-xl-sft",
        "revision": "d06de46b4622f781cf07f4a013a67d591ca52819",
        "desc": "Diffusion Transformer 4B (Supervised Fine-Tuned with CFG)",
        "expected_size_gb": 19.95,
        "required_files": ["config.json", "model.safetensors.index.json"],
        "allow_patterns": None,
    },
    "acestep-5Hz-lm-4B": {
        "repo_id": "ACE-Step/acestep-5Hz-lm-4B",
        "revision": "0a3ec94b557aea7d508da38b31cfe7341f6ff737",
        "desc": "Language Model Planner 4B (Qwen3-based music planning)",
        "expected_size_gb": 8.42,
        "required_files": ["config.json", "model.safetensors.index.json"],
        "allow_patterns": None,
    },
    "vae": {
        "repo_id": "ACE-Step/Ace-Step1.5",
        "revision": "19671f406d603126926c1b7e2adc169acbcade22",
        "desc": "Shared Oobleck Audio Autoencoder (48kHz stereo)",
        "expected_size_gb": 0.50,
        "required_files": ["config.json", "diffusion_pytorch_model.safetensors"],
        "allow_patterns": ["vae/*"],
        "subfolder": "vae",
    },
    "Qwen3-Embedding-0.6B": {
        "repo_id": "ACE-Step/Ace-Step1.5",
        "revision": "19671f406d603126926c1b7e2adc169acbcade22",
        "desc": "Shared text conditioning embedding model",
        "expected_size_gb": 1.00,
        "required_files": ["config.json", "model.safetensors"],
        "allow_patterns": ["Qwen3-Embedding-0.6B/*"],
        "subfolder": "Qwen3-Embedding-0.6B",
    },
}


def check_model_integrity(target_dir: Path, required_files: List[str]) -> Tuple[bool, List[str]]:
    """Check if all required files exist in target directory."""
    if not target_dir.exists():
        return False, ["directory_missing"]

    missing = []
    for req in required_files:
        if not (target_dir / req).exists():
            missing.append(req)

    # Verify at least one weight file exists
    weight_files = (
        list(target_dir.glob("*.safetensors"))
        + list(target_dir.glob("*.bin"))
        + list(target_dir.glob("*.pt"))
    )
    if not weight_files:
        missing.append("weights(*.safetensors|*.bin|*.pt)")

    return (len(missing) == 0), missing


def get_disk_free_gb(target_path: Path) -> float:
    """Get available disk space in GB."""
    try:
        usage = shutil.disk_usage(str(target_path.resolve()))
        return round(usage.free / (1024 ** 3), 1)
    except Exception:
        return 999.0


def download_models(
    checkpoints_dir: Path,
    model_keys: Optional[List[str]] = None,
    dry_run: bool = False,
    force: bool = False,
    token: Optional[str] = None,
) -> bool:
    """Download required checkpoints with cache verification and space checks."""
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    keys_to_process = model_keys or list(PINNED_CHECKPOINTS.keys())

    print("=" * 75)
    print("      music-gpu-lab :: Targeted Model Checkpoint Downloader      ")
    print(f"Destination Directory : {checkpoints_dir.resolve()}")
    print("=" * 75)

    total_expected_gb = sum(PINNED_CHECKPOINTS[k]["expected_size_gb"] for k in keys_to_process)
    pending_download_gb = 0.0
    cache_hits = []
    needs_download = []

    for key in keys_to_process:
        spec = PINNED_CHECKPOINTS[key]
        dest_dir = checkpoints_dir / key
        valid, missing = check_model_integrity(dest_dir, spec["required_files"])

        if valid and not force:
            cache_hits.append((key, dest_dir))
        else:
            needs_download.append((key, spec, dest_dir, missing))
            pending_download_gb += spec["expected_size_gb"]

    free_gb = get_disk_free_gb(checkpoints_dir)

    print(f"\n[Storage Audit]")
    print(f"  Total Models Cataloged  : {len(keys_to_process)} ({round(total_expected_gb, 2)} GB total)")
    print(f"  Cached / Already Valid  : {len(cache_hits)} models")
    print(f"  Pending Download        : {len(needs_download)} models ({round(pending_download_gb, 2)} GB)")
    print(f"  Available Disk Space    : {free_gb} GB")

    for key, path in cache_hits:
        spec = PINNED_CHECKPOINTS[key]
        print(f"  -> [CACHE HIT]  {key:<22} : Verified at {path} ({spec['repo_id']} @ {spec['revision'][:10]})")

    for key, spec, path, missing in needs_download:
        reasons = f"missing {', '.join(missing)}" if missing else "forced"
        print(f"  -> [NEED DOWNLOAD] {key:<19} : ~{spec['expected_size_gb']} GB ({reasons})")
        print(f"        Repo: {spec['repo_id']} @ {spec['revision'][:10]}")

    if needs_download and free_gb < (pending_download_gb + 5.0):
        print(f"\n[WARNING] Low disk space! Need ~{pending_download_gb + 5.0} GB but only {free_gb} GB free.")

    if dry_run:
        print("\n[DRY RUN] Inspection complete. No files were downloaded.")
        return True

    if not needs_download:
        print(f"\n[OK] All {len(keys_to_process)} checkpoints are present and verified in cache.")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n[ERROR] 'huggingface_hub' is required. Run 'pip install huggingface_hub' first.", file=sys.stderr)
        return False

    download_start_time = time.time()
    success = True

    for key, spec, dest_dir, _ in needs_download:
        print(f"\n>>> Downloading {key} (~{spec['expected_size_gb']} GB)...")
        print(f"    From: {spec['repo_id']} (rev: {spec['revision']})")
        item_start = time.time()

        try:
            subfolder = spec.get("subfolder")
            if subfolder:
                # For models packed inside subdirectories (e.g. Ace-Step1.5/vae)
                temp_parent = checkpoints_dir / ".tmp_download"
                temp_parent.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    local_dir=str(temp_parent),
                    allow_patterns=spec.get("allow_patterns"),
                    token=token,
                )
                source_sub = temp_parent / subfolder
                if source_sub.exists():
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.move(str(source_sub), str(dest_dir))
                shutil.rmtree(temp_parent, ignore_errors=True)
            else:
                snapshot_download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    local_dir=str(dest_dir),
                    allow_patterns=spec.get("allow_patterns"),
                    token=token,
                )

            valid, missing = check_model_integrity(dest_dir, spec["required_files"])
            item_elapsed = round(time.time() - item_start, 1)
            if valid:
                print(f"[OK] {key} downloaded and verified in {item_elapsed}s.")
            else:
                print(f"[ERROR] {key} downloaded but failed integrity check. Missing: {missing}", file=sys.stderr)
                success = False

        except Exception as e:
            print(f"[FAIL] Download failed for {key}: {e}", file=sys.stderr)
            success = False

    total_elapsed = round(time.time() - download_start_time, 1)
    print("\n" + "=" * 75)
    if success:
        print(f"[SUCCESS] Checkpoint download finished in {total_elapsed}s.")
    else:
        print(f"[FAIL] One or more checkpoints failed to download properly.", file=sys.stderr)
    print("=" * 75)

    return success


def parse_args():
    parser = argparse.ArgumentParser(description="Targeted Pinned ACE-Step 1.5 Checkpoint Downloader")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(os.environ.get("CHECKPOINTS_DIR", "./checkpoints")),
        help="Target root directory for checkpoints (default: ./checkpoints or $CHECKPOINTS_DIR)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(PINNED_CHECKPOINTS.keys()),
        default=None,
        help="Specific models to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect cache hits, estimated sizes, and disk space without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download checkpoints even if already verified in cache",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Optional Hugging Face token",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ok = download_models(
        checkpoints_dir=args.checkpoints_dir,
        model_keys=args.models,
        dry_run=args.dry_run,
        force=args.force,
        token=args.token,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
