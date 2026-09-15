#!/usr/bin/env python3
"""
Benchmark Indexer for music-gpu-lab.
Aggregates candidate outputs, timings, VRAM, and RTF into Markdown, CSV, and JSON indexes.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def index_benchmark(benchmark_dir: Path) -> bool:
    """Scan all candidate metadata files within benchmark_dir and compile index."""
    if not benchmark_dir.exists():
        print(f"[ERROR] Benchmark directory does not exist: {benchmark_dir}", file=sys.stderr)
        return False

    records: List[Dict[str, Any]] = []

    # Search for all generation_metadata.json files
    meta_files = sorted(benchmark_dir.glob("**/generation_metadata.json"))
    repo_root = Path(__file__).resolve().parent.parent

    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)

            preset = data.get("preset", {}).get("name", "unknown")
            mode = data.get("preset", {}).get("mode", "unknown")
            params = data.get("parameters", {})
            seed = params.get("seed", -1)
            duration = params.get("duration_seconds", 0.0)
            bpm = params.get("bpm", 0)
            cfg = params.get("guidance_scale", 7.0)
            perf = data.get("performance", {})
            elapsed = perf.get("elapsed_seconds", 0.0)
            rtf = perf.get("real_time_factor", 0.0)
            vram = perf.get("peak_vram_allocated_gb", 0.0)
            audio_name = data.get("artifacts", {}).get("audio_file", "None")
            audio_rel_path = (mf.parent / audio_name)
            try:
                audio_str = str(audio_rel_path.relative_to(benchmark_dir))
            except ValueError:
                audio_str = audio_name

            records.append({
                "preset": preset,
                "mode": mode,
                "seed": seed,
                "duration_seconds": duration,
                "bpm": bpm,
                "guidance_scale": cfg,
                "elapsed_seconds": elapsed,
                "real_time_factor": rtf,
                "peak_vram_gb": vram,
                "audio_path": audio_str,
                "success": data.get("success", False),
            })
        except Exception as e:
            print(f"[WARN] Failed to parse {mf}: {e}", file=sys.stderr)

    if not records:
        print(f"[WARN] No generation metadata files found in {benchmark_dir}")
        return False

    # 1. Write benchmark_index.json
    with open(benchmark_dir / "benchmark_index.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_dir": str(benchmark_dir.resolve()),
            "total_tracks": len(records),
            "tracks": records,
        }, f, indent=2)

    # 2. Write benchmark_index.csv
    csv_path = benchmark_dir / "benchmark_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "preset", "mode", "seed", "duration_seconds", "bpm",
            "guidance_scale", "elapsed_seconds", "real_time_factor",
            "peak_vram_gb", "audio_path", "success",
        ])
        writer.writeheader()
        writer.writerows(records)

    # 3. Write benchmark_index.md
    md_lines = [
        "# First-Hour Benchmark Suite Summary",
        f"\n**Total Evaluations:** {len(records)} tracks | **Status:** All runs cataloged\n",
        "| Style / Preset | Mode | Seed | Dur (s) | BPM | CFG | Wall Time | RTF | Peak VRAM | Audio File |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in records:
        md_lines.append(
            f"| `{r['preset']}` | `{r['mode']}` | `{r['seed']}` | {r['duration_seconds']}s | {r['bpm']} | {r['guidance_scale']} | {r['elapsed_seconds']}s | {r['real_time_factor']}x | {r['peak_vram_gb']} GB | `{r['audio_path']}` |"
        )
    md_lines.append("")

    with open(benchmark_dir / "benchmark_index.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[OK] Generated index files in {benchmark_dir.resolve()}:")
    print(f"  - benchmark_index.md")
    print(f"  - benchmark_index.csv")
    print(f"  - benchmark_index.json")
    return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark Suite Index Generator")
    parser.add_argument("benchmark_dir", type=Path, help="Path to benchmark run directory")
    args = parser.parse_args()
    ok = index_benchmark(args.benchmark_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
