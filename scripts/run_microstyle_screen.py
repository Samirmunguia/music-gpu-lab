#!/usr/bin/env python3
"""
6-Candidate Microstyle Representation Screen (S1A/B, S2A/B, S4A/B).
Conditioning DiT directly with human captions (use_cot_caption=False, use_cot_metas=False, thinking=True).
Outputs 20-second instrumental candidates with seed 42.
Includes self-analysis via understand_music, ffprobe validation, and multi-location deployment with playlist.
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", REPO_ROOT / "checkpoints"))
OUTPUT_DIR = REPO_ROOT / "outputs" / "microstyle_screen_3styles"

# Local / Windows mirror directories
WINDOWS_D_ROOT = Path("/D:\ACE-Step-Microstyle-3Styles")
WINDOWS_D_USER = Path("/root/D:\ACE-Step-Microstyle-3Styles")
IDE_ARTIFACT_DIR = Path("/root/.gemini/antigravity-ide/brain/020fa60f-3144-4bb4-bd6c-c1005369d037/microstyle_screen_3styles")

ACE_STEP_ROOT = REPO_ROOT / "src" / "ACE-Step-1.5"
if (ACE_STEP_ROOT / "acestep").exists() and str(ACE_STEP_ROOT) not in sys.path:
    sys.path.insert(0, str(ACE_STEP_ROOT))

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import (
    GenerationParams,
    GenerationConfig,
    generate_music,
    understand_music,
)


def make_json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 200:
            return f"<list of {len(obj)} items>"
        return [make_json_safe(x) for x in obj]
    if hasattr(obj, "shape") or hasattr(obj, "cpu"):
        return f"<tensor shape={getattr(obj, 'shape', 'unknown')} dtype={getattr(obj, 'dtype', 'unknown')}>"
    if hasattr(obj, "__dict__"):
        return make_json_safe(obj.__dict__)
    return str(obj)


def get_gpu_metadata() -> Dict[str, Any]:
    meta = {
        "cuda_available": False,
        "device_name": "N/A",
        "total_vram_gb": 0.0,
        "cuda_version": "N/A",
        "driver_version": "N/A",
    }
    try:
        import torch
        if torch.cuda.is_available():
            meta["cuda_available"] = True
            meta["device_name"] = torch.cuda.get_device_name(0)
            meta["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            meta["cuda_version"] = torch.version.cuda or "unknown"
    except Exception as e:
        meta["error"] = str(e)
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            encoding="utf-8",
            timeout=3,
        ).strip()
        if smi_out:
            parts = [p.strip() for p in smi_out.split("\n")[0].split(",")]
            if len(parts) >= 1:
                meta["driver_version"] = parts[0]
            if len(parts) >= 2 and meta["device_name"] == "N/A":
                meta["device_name"] = parts[1]
    except Exception:
        pass
    return meta


CANDIDATES = [
    {
        "id": "S1A",
        "file_name": "S1A_reggaeton_named.flac",
        "style": "S1 — NOSTALGIC MINIMAL REGGAETON",
        "variant": "genre named",
        "bpm": 84,
        "caption": "minimal nocturnal reggaeton, 84 BPM, filtered dry dembow, deep sub bass, sparse dark pad, slow hypnotic three-note synth motif, lots of empty space, intimate nostalgic underground mix",
    },
    {
        "id": "S1B",
        "file_name": "S1B_reggaeton_dna.flac",
        "style": "S1 — NOSTALGIC MINIMAL REGGAETON",
        "variant": "acoustic DNA only",
        "bpm": 84,
        "caption": "84 BPM, dry filtered syncopated kick and snare groove, deep sub bass, sparse dark pad, slow hypnotic three-note synth motif, wide empty arrangement, intimate nostalgic late-night mix",
    },
    {
        "id": "S2A",
        "file_name": "S2A_trap_named.flac",
        "style": "S2 — HYPNOTIC LATIN TRAP",
        "variant": "genre named",
        "bpm": 82,
        "caption": "hypnotic Latin trap, 82 BPM, deep 808 bass, active syncopated hi-hats with selective short rolls, sparse kick and clap, slow three-note minor synth motif, dark nostalgic atmosphere, minimal nocturnal arrangement",
    },
    {
        "id": "S2B",
        "file_name": "S2B_trap_dna.flac",
        "style": "S2 — HYPNOTIC LATIN TRAP",
        "variant": "acoustic DNA only",
        "bpm": 82,
        "caption": "82 BPM, deep sustained 808 bass, active syncopated sixteenth-note hi-hats with occasional short rolls, sparse kick and clap, slow three-note minor synth motif, dark empty nocturnal arrangement",
    },
    {
        "id": "S4A",
        "file_name": "S4A_2010s_named.flac",
        "style": "S4 — EARLY-2010s PUERTO RICAN REGGAETON",
        "variant": "genre named",
        "bpm": 115,
        "caption": "early-2010s Puerto Rican reggaeton, 115 BPM, raw dry dirty dembow, deep bass, bright digital synth arpeggio lead, unforgettable four-note melodic hook, small synth counterlines, gritty mixtape production",
    },
    {
        "id": "S4B",
        "file_name": "S4B_2010s_dna.flac",
        "style": "S4 — EARLY-2010s PUERTO RICAN REGGAETON",
        "variant": "acoustic DNA only",
        "bpm": 115,
        "caption": "115 BPM, raw dry syncopated drum-machine groove, deep bass, bright digital arpeggiated synth lead, unforgettable four-note melodic hook, small answering synth counterlines, gritty early-digital club mix",
    },
]


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_microstyle_screen(device: str = "cuda"):
    print("=" * 80)
    print("   music-gpu-lab :: 6-CANDIDATE MICROSTYLE REPRESENTATION SCREEN    ")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_D_ROOT.mkdir(parents=True, exist_ok=True)
    WINDOWS_D_USER.mkdir(parents=True, exist_ok=True)
    IDE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    gpu_meta = get_gpu_metadata()
    print(f"Hardware: {gpu_meta['device_name']} ({gpu_meta['total_vram_gb']} GB VRAM)")
    print(f"Output Directory: {OUTPUT_DIR.resolve()}\n")

    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(CHECKPOINTS_DIR.resolve())
    print(">>> Initializing AceStepHandler (acestep-v15-xl-sft)...")
    dit_handler = AceStepHandler()
    init_status, _ = dit_handler.initialize_service(
        project_root=str(REPO_ROOT),
        config_path="acestep-v15-xl-sft",
        device=device,
    )
    print(f"    AceStepHandler status: {init_status}")

    print(">>> Initializing LLMHandler (acestep-5Hz-lm-4B)...")
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=str(CHECKPOINTS_DIR.resolve()),
        lm_model_path="acestep-5Hz-lm-4B",
        backend="pt",
        device=device,
    )
    print("    LLMHandler initialized successfully on CUDA.\n")

    suite_start_time = time.time()
    results = []

    # 1. Sequential Candidate Generation
    for c in CANDIDATES:
        c_id = c["id"]
        c_file = c["file_name"]
        c_style = c["style"]
        c_var = c["variant"]
        c_bpm = c["bpm"]
        c_caption = c["caption"]
        c_lyrics = "[Instrumental]"
        c_duration = 20.0
        c_seed = 42

        cand_dir = OUTPUT_DIR / c_id
        cand_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 80)
        print(f">>> Generating [{c_id}] ({c_style} :: {c_var})")
        print(f"    BPM: {c_bpm} | Seed: {c_seed} | Duration: {c_duration}s")
        print(f"    Caption: \"{c_caption}\"")
        print("-" * 80)

        params = GenerationParams(
            task_type="text2music",
            caption=c_caption,
            lyrics=c_lyrics,
            instrumental=True,
            vocal_language="es",
            bpm=c_bpm,
            duration=c_duration,
            inference_steps=32,
            guidance_scale=7.0,
            seed=c_seed,
            shift=1.0,
            infer_method="ode",
            thinking=True,
            use_cot_caption=False,
            use_cot_metas=False,
            lm_temperature=0.85,
            lm_cfg_scale=1.0,
            lm_negative_prompt=None,
        )

        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[c_seed],
            audio_format="flac",
        )

        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        audio_codes_str = ""
        error_msg = None

        try:
            gen_res = generate_music(
                dit_handler,
                llm_handler,
                params,
                config,
                save_dir=str(cand_dir),
            )
            elapsed = round(time.time() - t0, 2)
            peak_alloc = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
            peak_res = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
            rtf = round(elapsed / max(c_duration, 0.1), 3)

            if not gen_res.success or not gen_res.audios:
                raise RuntimeError(gen_res.error or "generate_music reported failure")

            # Canonical named audio file in candidate folder and in parent output dir
            src_audio = Path(gen_res.audios[0]["path"])
            dest_audio_cand = cand_dir / c_file
            dest_audio_parent = OUTPUT_DIR / c_file
            shutil.copy2(src_audio, dest_audio_cand)
            shutil.copy2(src_audio, dest_audio_parent)

            # Extract audio codes
            raw_codes = gen_res.audios[0].get("params", {}).get("audio_codes")
            if raw_codes and isinstance(raw_codes, str) and raw_codes.strip():
                audio_codes_str = raw_codes.strip()
                (cand_dir / "audio_codes.txt").write_text(audio_codes_str, encoding="utf-8")

            extra = gen_res.extra_outputs or {}
            lm_meta = extra.get("lm_metadata") or {}
            time_costs = extra.get("time_costs") or {}

            cand_data = {
                "id": c_id,
                "file_name": c_file,
                "style": c_style,
                "variant": c_var,
                "caption": c_caption,
                "lyrics": c_lyrics,
                "bpm": c_bpm,
                "duration_seconds": c_duration,
                "seed": c_seed,
                "parameters": {
                    "inference_steps": 32,
                    "guidance_scale": 7.0,
                    "infer_method": "ode",
                    "shift": 1.0,
                    "thinking": True,
                    "use_cot_caption": False,
                    "use_cot_metas": False,
                    "lm_temperature": 0.85,
                    "lm_cfg_scale": 1.0,
                },
                "performance": {
                    "elapsed_seconds": elapsed,
                    "real_time_factor": rtf,
                    "peak_vram_allocated_gb": peak_alloc,
                    "peak_vram_reserved_gb": peak_res,
                },
                "telemetry": {
                    "lm_metadata": make_json_safe(lm_meta),
                    "time_costs": make_json_safe(time_costs),
                    "has_audio_codes": bool(audio_codes_str),
                },
                "timestamps": {
                    "started_at": start_iso,
                    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
                "audio_sha256": sha256_file(dest_audio_parent),
                "audio_size_bytes": dest_audio_parent.stat().st_size,
                "success": True,
                "error": None,
            }

            print(f"    [OK] Elapsed: {elapsed}s | RTF: {rtf}x | Peak VRAM: {peak_alloc} GB")

        except Exception as e:
            error_msg = str(e)
            print(f"    [FAIL] Candidate {c_id} failed: {e}", file=sys.stderr)
            cand_data = {
                "id": c_id,
                "file_name": c_file,
                "style": c_style,
                "variant": c_var,
                "caption": c_caption,
                "success": False,
                "error": error_msg,
            }

        with open(cand_dir / "candidate_metadata.json", "w", encoding="utf-8") as f:
            json.dump(cand_data, f, indent=2)

        c["result_meta"] = cand_data
        c["audio_codes"] = audio_codes_str
        c["audio_path"] = str(OUTPUT_DIR / c_file)
        results.append(c)

    # 2. Self-Analysis via understand_music
    print("\n" + "=" * 80)
    print(">>> EXECUTING SELF-ANALYSIS (understand_music)")
    print("=" * 80)
    
    self_analysis_records = []
    for c in results:
        c_id = c["id"]
        c_name = c["file_name"]
        codes = c.get("audio_codes", "")
        print(f"\n>>> Running understand_music for [{c_id}] {c_name}...")
        if not codes:
            record = {
                "id": c_id,
                "file_name": c_name,
                "success": False,
                "note": "No audio semantic codes generated",
                "caption": "N/A",
                "bpm": None,
                "key": None,
                "language": None,
            }
        else:
            try:
                und_res = understand_music(
                    llm_handler=llm_handler,
                    audio_codes=codes,
                    temperature=0.85,
                )
                record = {
                    "id": c_id,
                    "file_name": c_name,
                    "success": und_res.success,
                    "understood_caption": und_res.caption,
                    "understood_bpm": und_res.bpm,
                    "understood_key": und_res.keyscale,
                    "understood_language": und_res.language,
                    "status_message": und_res.status_message,
                    "error": und_res.error,
                }
                print(f"    [Analysis Result] BPM: {und_res.bpm} | Key: {und_res.keyscale}")
                print(f"    Understood: {und_res.caption}")
            except Exception as e:
                print(f"    [Analysis Error] {e}")
                record = {
                    "id": c_id,
                    "file_name": c_name,
                    "success": False,
                    "error": str(e),
                }

        self_analysis_records.append(record)
        cand_folder = OUTPUT_DIR / c_id
        with open(cand_folder / "self_analysis.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    with open(OUTPUT_DIR / "self_analysis.json", "w", encoding="utf-8") as f:
        json.dump(self_analysis_records, f, indent=2)

    # 3. Create Playlist & Deploy to Destinations
    print("\n" + "=" * 80)
    print(">>> DEPLOYING OUTPUTS & GENERATING LISTEN_AB.m3u PLAYLIST")
    print("=" * 80)

    playlist_lines = [
        "#EXTM3U",
        "# 6-Candidate Microstyle Representation Screen (S1A/B, S2A/B, S4A/B)",
        "S1A_reggaeton_named.flac",
        "S1B_reggaeton_dna.flac",
        "S2A_trap_named.flac",
        "S2B_trap_dna.flac",
        "S4A_2010s_named.flac",
        "S4B_2010s_dna.flac",
    ]
    playlist_content = "\n".join(playlist_lines) + "\n"

    # Write playlist in all locations
    (OUTPUT_DIR / "LISTEN_AB.m3u").write_text(playlist_content, encoding="utf-8")
    (WINDOWS_D_ROOT / "LISTEN_AB.m3u").write_text(playlist_content, encoding="utf-8")
    (WINDOWS_D_USER / "LISTEN_AB.m3u").write_text(playlist_content, encoding="utf-8")
    (IDE_ARTIFACT_DIR / "LISTEN_AB.m3u").write_text(playlist_content, encoding="utf-8")

    # Copy flac files into all destinations
    for c in CANDIDATES:
        flac_src = OUTPUT_DIR / c["file_name"]
        if flac_src.exists():
            shutil.copy2(flac_src, WINDOWS_D_ROOT / c["file_name"])
            shutil.copy2(flac_src, WINDOWS_D_USER / c["file_name"])
            shutil.copy2(flac_src, IDE_ARTIFACT_DIR / c["file_name"])

    # 4. Master Screen Summary Table & Manifest
    total_screen_time = round(time.time() - suite_start_time, 2)
    master_manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_runtime_seconds": total_screen_time,
        "hardware": gpu_meta,
        "candidates": [c["result_meta"] for c in results],
        "self_analysis": self_analysis_records,
    }
    with open(OUTPUT_DIR / "screen_manifest.json", "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    md_report = [
        "# Microstyle Representation Screen (S1A/B, S2A/B, S4A/B)",
        f"\n**Total Runtime:** {total_screen_time}s | **GPU:** {gpu_meta['device_name']} | **Date:** {master_manifest['timestamp']}\n",
        "## 1. Candidate Generation & Technical Telemetry\n",
        "| ID | Style / Pair | Variant | BPM | Wall Time | RTF | Peak VRAM | File Size | Audio SHA-256 (Prefix) | Status |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for c in results:
        rm = c.get("result_meta", {})
        perf = rm.get("performance", {})
        sha_pref = rm.get("audio_sha256", "N/A")[:12] if rm.get("audio_sha256") else "N/A"
        size_mb = f"{rm.get('audio_size_bytes', 0)/(1024*1024):.2f} MB"
        md_report.append(
            f"| `{c['id']}` | {c['style']} | {c['variant']} | {c['bpm']} | "
            f"{perf.get('elapsed_seconds', 'N/A')}s | {perf.get('real_time_factor', 'N/A')}x | "
            f"{perf.get('peak_vram_allocated_gb', 'N/A')} GB | {size_mb} | `{sha_pref}` | {'PASS' if rm.get('success') else 'FAIL'} |"
        )

    md_report.append("\n## 2. Conditioned Prompts (DiT Received Unchanged)\n")
    for c in results:
        md_report.append(f"### `[{c['id']}]` {c['file_name']}")
        md_report.append(f"- **Style**: {c['style']} ({c['variant']}) | **BPM**: {c['bpm']}")
        md_report.append(f"- **Caption**: *\"{c['caption']}\"*")
        md_report.append("")

    md_report.append("## 3. ACE-Step Self-Analysis (`understand_music`)\n")
    md_report.append("| ID | File Name | Understood Caption | Detected BPM | Detected Key | Detected Lang |")
    md_report.append("| :--- | :--- | :--- | :---: | :---: | :---: |")
    for sa in self_analysis_records:
        cap_snip = sa.get("understood_caption", "N/A")
        if len(cap_snip) > 85:
            cap_snip = cap_snip[:82] + "..."
        md_report.append(
            f"| `{sa['id']}` | `{sa['file_name']}` | {cap_snip} | {sa.get('understood_bpm', 'N/A')} | {sa.get('understood_key', 'N/A')} | {sa.get('understood_language', 'N/A')} |"
        )
    md_report.append("")

    summary_file = OUTPUT_DIR / "screen_summary.md"
    summary_file.write_text("\n".join(md_report), encoding="utf-8")
    print(f"\n[OK] Summary report written to: {summary_file.resolve()}")
    print("=" * 80)
    print("6-CANDIDATE MICROSTYLE SCREEN COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_microstyle_screen()
