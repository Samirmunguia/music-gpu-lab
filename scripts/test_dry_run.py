#!/usr/bin/env python3
"""
Test harness for music-gpu-lab.
Validates preset schemas, CLI arguments, live smoke test dry-run,
in-memory batch generation with manifests, and benchmark suite indexing
without requiring an NVIDIA GPU or physical model weights.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = REPO_ROOT / "presets"
EXPECTED_PRESETS = ["reggaeton_nocturno", "dembow_dominicano", "reggaeton_2010s"]
EXPECTED_MODES = ["instrumental", "male-vocal"]


def test_presets_schema():
    """Verify all 3 presets exist and adhere to expected structure."""
    print("\n[TEST] 1. Validating Preset Schemas...")
    for preset_name in EXPECTED_PRESETS:
        preset_file = PRESETS_DIR / f"{preset_name}.json"
        assert preset_file.exists(), f"Preset file missing: {preset_file}"

        with open(preset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("name") == preset_name, f"Mismatched name in {preset_file}"
        assert "bpm" in data and isinstance(data["bpm"], int), f"Missing integer bpm in {preset_file}"
        assert "default_duration" in data, f"Missing default_duration in {preset_file}"
        assert "default_seed" in data, f"Missing default_seed in {preset_file}"
        assert "modes" in data, f"Missing modes object in {preset_file}"

        for mode in EXPECTED_MODES:
            assert mode in data["modes"], f"Mode '{mode}' missing from {preset_file}"
            mode_data = data["modes"][mode]
            assert "caption" in mode_data and len(mode_data["caption"]) > 10, f"Invalid caption in {preset_file} [{mode}]"
            assert "lyrics" in mode_data and len(mode_data["lyrics"]) > 0, f"Invalid lyrics in {preset_file} [{mode}]"

        print(f"  [OK] Preset '{preset_name}' (BPM: {data['bpm']}, Modes: {list(data['modes'].keys())})")


def test_download_checkpoints_dry_run():
    """Verify checkpoint downloader dry-run and pinned revision inspection."""
    print("\n[TEST] 2. Validating Checkpoint Downloader (Pinned Revisions & Dry Run)...")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "download_checkpoints.py"),
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"download_checkpoints.py failed: {res.stderr}"
    assert "acestep-v15-xl-sft" in res.stdout
    assert "acestep-5Hz-lm-4B" in res.stdout
    assert "d06de46b46" in res.stdout  # Pinned revision prefix
    assert "0a3ec94b55" in res.stdout  # Pinned revision prefix
    assert "19671f406d" in res.stdout  # Pinned VAE revision prefix
    print("  [OK] Pinned revisions and storage audit verified.")


def test_smoke_dry_run():
    """Verify smoke test command flag and dry-run execution."""
    print("\n[TEST] 3. Validating Smoke Test Flag (--smoke)...")
    test_temp_dir = Path(tempfile.mkdtemp(prefix="music_gpu_lab_smoke_"))

    try:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate.py"),
            "--smoke",
            "--output-dir", str(test_temp_dir),
            "--dry-run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Smoke dry-run failed:\n{res.stderr}\n{res.stdout}"

        meta_file = test_temp_dir / "generation_metadata.json"
        assert meta_file.exists(), "Missing smoke generation_metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["smoke"] is True
        assert meta["preset"]["name"] == "smoke"
        assert meta["parameters"]["duration_seconds"] == 5.0
        assert meta["parameters"]["inference_steps"] == 8
        print("  [OK] Smoke test pipeline flag and parameters verified.")

    finally:
        shutil.rmtree(test_temp_dir, ignore_errors=True)


def test_batch_generation_with_manifest():
    """Verify batch generation (--seeds N), candidate directories, and batch manifest."""
    print("\n[TEST] 4. Validating Batch Generation (--seeds 3) & Manifest...")
    test_temp_dir = Path(tempfile.mkdtemp(prefix="music_gpu_lab_batch_"))

    try:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate.py"),
            "ace-step",
            "dembow_dominicano",
            "instrumental",
            "--seed", "100",
            "--seeds", "3",
            "--duration", "15.0",
            "--output-dir", str(test_temp_dir),
            "--dry-run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Batch generation failed:\n{res.stderr}\n{res.stdout}"

        # Check candidate folders
        expected_seeds = [100, 101, 102]
        for s in expected_seeds:
            cand_dir = test_temp_dir / f"seed_{s}"
            assert cand_dir.exists(), f"Candidate directory missing: {cand_dir}"
            assert (cand_dir / "generation_metadata.json").exists(), f"Missing metadata in {cand_dir}"
            assert (cand_dir / "prompt.txt").exists(), f"Missing prompt.txt in {cand_dir}"

            with open(cand_dir / "generation_metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["parameters"]["seed"] == s
            assert "real_time_factor" in meta["performance"]
            assert meta["git_commit_sha"] != ""
            assert meta["upstream_commit_sha"] == "ca1e85fe9430179831e6bc6be790c332190a3866"

        # Check batch manifest files
        manifest_json = test_temp_dir / "batch_manifest.json"
        manifest_md = test_temp_dir / "batch_manifest.md"
        assert manifest_json.exists(), "Missing batch_manifest.json"
        assert manifest_md.exists(), "Missing batch_manifest.md"

        with open(manifest_json, "r", encoding="utf-8") as f:
            man_data = json.load(f)
        assert man_data["total_candidates"] == 3
        assert len(man_data["candidates"]) == 3

        print(f"  [OK] Generated 3 candidates with manifests (batch_manifest.json & .md)")

    finally:
        shutil.rmtree(test_temp_dir, ignore_errors=True)


def test_benchmark_suite_dry_run_and_indexing():
    """Verify entire benchmark suite dry-run and index compilation."""
    print("\n[TEST] 5. Validating Benchmark Suite Indexing (scripts/index_benchmark.py)...")
    test_temp_dir = Path(tempfile.mkdtemp(prefix="music_gpu_lab_bench_"))

    try:
        # Simulate benchmark outputs
        for style in EXPECTED_PRESETS:
            for mode in ["instrumental", "male-vocal"]:
                out_dir = test_temp_dir / f"{style}_{mode}"
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "generate.py"),
                    "ace-step",
                    style,
                    mode,
                    "--seeds", "2",
                    "--duration", "10",
                    "--output-dir", str(out_dir),
                    "--dry-run",
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                assert res.returncode == 0, f"Generate failed on {style}_{mode}: {res.stderr}"

        # Run indexer
        idx_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "index_benchmark.py"),
            str(test_temp_dir),
        ]
        res = subprocess.run(idx_cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Index benchmark failed: {res.stderr}"

        # Verify index files
        md_file = test_temp_dir / "benchmark_index.md"
        csv_file = test_temp_dir / "benchmark_index.csv"
        json_file = test_temp_dir / "benchmark_index.json"

        assert md_file.exists(), "Missing benchmark_index.md"
        assert csv_file.exists(), "Missing benchmark_index.csv"
        assert json_file.exists(), "Missing benchmark_index.json"

        with open(json_file, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        assert idx_data["total_tracks"] == 12  # 3 styles * 2 modes * 2 seeds = 12 tracks

        with open(csv_file, "r", encoding="utf-8") as f:
            csv_reader = list(csv.DictReader(f))
        assert len(csv_reader) == 12

        print(f"  [OK] Indexer successfully cataloged 12 tracks into MD, CSV, and JSON.")

    finally:
        shutil.rmtree(test_temp_dir, ignore_errors=True)


def test_cli_overrides():
    """Verify parameter overrides in metadata."""
    print("\n[TEST] 6. Validating CLI Parameter Overrides...")
    test_temp_dir = Path(tempfile.mkdtemp(prefix="music_gpu_lab_override_"))

    try:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate.py"),
            "ace-step",
            "reggaeton_nocturno",
            "instrumental",
            "--bpm", "102",
            "--duration", "18.5",
            "--seed", "9999",
            "--guidance-scale", "8.5",
            "--steps", "45",
            "--output-dir", str(test_temp_dir),
            "--dry-run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Override test failed: {res.stderr}"

        with open(test_temp_dir / "generation_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        params = meta["parameters"]
        assert params["bpm"] == 102
        assert params["duration_seconds"] == 18.5
        assert params["seed"] == 9999
        assert params["guidance_scale"] == 8.5
        assert params["inference_steps"] == 45
        print("  [OK] CLI overrides (BPM=102, Duration=18.5s, Seed=9999, CFG=8.5, Steps=45) verified.")

    finally:
        shutil.rmtree(test_temp_dir, ignore_errors=True)


def test_verify_dependencies():
    """Verify dependency audit script runs without uncaught exceptions."""
    print("\n[TEST] 7. Validating Dependency Auditor (scripts/verify_dependencies.py)...")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_dependencies.py"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Without --strict, it should exit 0 and print the audit table
    assert res.returncode == 0, f"verify_dependencies.py failed: {res.stderr}"
    assert "Dependency Audit" in res.stdout
    assert "transformers" in res.stdout
    assert "torchao" in res.stdout
    assert "diffusers" in res.stdout
    print("  [OK] Dependency auditor executed and printed audit table.")


def main():
    print("=" * 75)
    print("           music-gpu-lab Offline Test Suite")
    print("=" * 75)

    try:
        test_presets_schema()
        test_download_checkpoints_dry_run()
        test_smoke_dry_run()
        test_batch_generation_with_manifest()
        test_benchmark_suite_dry_run_and_indexing()
        test_cli_overrides()
        test_verify_dependencies()
        print("\n" + "=" * 75)
        print("ALL 7 TEST SUITES PASSED SUCCESSFULLY! (Offline & Hardware-Agnostic)")
        print("=" * 75)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] Assertion Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
