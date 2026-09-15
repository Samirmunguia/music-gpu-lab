#!/usr/bin/env python3
"""
ACE-Step 1.5 Benchmark Generation Runner, Batch Processor, and Telemetry Logger.
Executes inference on acestep-v15-xl-sft (4B DiT) + acestep-5Hz-lm-4B.
Features in-memory model reuse across seeds, live smoke testing, and rich telemetry.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Pinned commit and model revisions
PINNED_UPSTREAM_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
PINNED_MODEL_REVISIONS = {
    "acestep-v15-xl-sft": "d06de46b4622f781cf07f4a013a67d591ca52819",
    "acestep-5Hz-lm-4B": "0a3ec94b557aea7d508da38b31cfe7341f6ff737",
    "vae": "19671f406d603126926c1b7e2adc169acbcade22",
    "Qwen3-Embedding-0.6B": "19671f406d603126926c1b7e2adc169acbcade22",
}


def get_lab_git_sha(repo_root: Path) -> str:
    """Retrieve current git commit SHA of music-gpu-lab repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        ).strip()
        return out or "untracked"
    except Exception:
        return "untracked"


def get_gpu_metadata() -> Dict[str, Any]:
    """Retrieve detailed NVIDIA GPU hardware and runtime telemetry."""
    meta: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "N/A",
        "total_vram_gb": 0.0,
        "cuda_version": "N/A",
        "driver_version": "N/A",
    }

    try:
        import torch
        if torch.cuda.is_available():
            meta["cuda_available"] = True
            meta["device_count"] = torch.cuda.device_count()
            meta["device_name"] = torch.cuda.get_device_name(0)
            meta["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            meta["cuda_version"] = torch.version.cuda or "unknown"
    except Exception as e:
        meta["torch_error"] = str(e)

    # Query nvidia-smi for driver version and exact device details
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
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


def load_preset(preset_arg: str, presets_dir: Path) -> Dict[str, Any]:
    """Load preset configuration from name or explicit file path."""
    if preset_arg == "smoke":
        return {
            "name": "smoke",
            "display_name": "Smoke Test",
            "description": "Minimal probe for live pipeline validation",
            "bpm": 120,
            "default_duration": 5.0,
            "default_seed": 42,
            "default_guidance_scale": 7.0,
            "default_inference_steps": 8,
            "vocal_language": "en",
            "modes": {
                "instrumental": {
                    "caption": "minimal electronic diagnostic probe tone, 120 bpm, clean mix, instrumental",
                    "lyrics": "[Instrumental]",
                    "instrumental": True,
                },
                "male-vocal": {
                    "caption": "minimal electronic diagnostic probe tone, 120 bpm, short male guide phrase",
                    "lyrics": "[Verse]\nTesting the live audio pipeline now\nSignal loud and clear",
                    "instrumental": False,
                },
            },
        }

    preset_path = Path(preset_arg)
    if not preset_path.exists():
        preset_path = presets_dir / f"{preset_arg}.json"

    if not preset_path.exists():
        raise FileNotFoundError(f"Preset '{preset_arg}' not found at {preset_path}")

    with open(preset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_batch_seeds(base_seed: int, count: int, seed_list: Optional[List[int]]) -> List[int]:
    """Resolve deterministic list of seeds for batch generation."""
    if seed_list and len(seed_list) > 0:
        return seed_list
    if count <= 1:
        return [base_seed]
    start = base_seed if base_seed >= 0 else 42
    return [start + i for i in range(count)]


def initialize_ace_step_pipeline(checkpoints_dir: Path, device: str) -> Tuple[Any, Any]:
    """Initialize AceStepHandler and LLMHandler once in memory."""
    import torch
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA is not available but device='cuda' was requested.")

    ace_step_root = None
    possible_roots = [
        Path(__file__).resolve().parent.parent / "src" / "ACE-Step-1.5",
        Path(__file__).resolve().parent.parent / "ACE-Step-1.5",
        Path.cwd() / "src" / "ACE-Step-1.5",
        Path.cwd(),
    ]
    for r in possible_roots:
        if (r / "acestep").exists():
            ace_step_root = r
            if str(r) not in sys.path:
                sys.path.insert(0, str(r))
            break

    try:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
    except ImportError as e:
        raise ImportError(
            f"Failed to import ACE-Step 1.5: {e}. "
            "Ensure ACE-Step-1.5 is cloned and 'pip install -e src/ACE-Step-1.5' was run."
        )

    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints_dir.resolve())

    print(f">>> Initializing AceStepHandler (acestep-v15-xl-sft) on {device}...")
    dit_handler = AceStepHandler()
    init_status, _ = dit_handler.initialize_service(
        project_root=str(ace_step_root or Path.cwd()),
        config_path="acestep-v15-xl-sft",
        device=device,
    )
    print(f"    AceStepHandler status: {init_status}")

    print(f">>> Initializing LLMHandler (acestep-5Hz-lm-4B) on {device}...")
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=str(checkpoints_dir.resolve()),
        lm_model_path="acestep-5Hz-lm-4B",
        backend="pt",
        device=device,
    )
    return dit_handler, llm_handler


def run_single_candidate(
    dit_handler: Any,
    llm_handler: Any,
    caption: str,
    lyrics: str,
    is_instrumental: bool,
    vocal_language: str,
    bpm: int,
    duration: float,
    seed: int,
    guidance_scale: float,
    inference_steps: int,
    device: str,
    candidate_dir: Path,
    dry_run: bool,
    log_lines: list,
) -> Dict[str, Any]:
    """Execute generation for a single seed candidate, writing all artifacts."""
    candidate_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    if dry_run:
        # Dry run simulation: synthetic validation audio
        audio_dest = candidate_dir / "audio.wav"
        try:
            import wave
            import struct
            with wave.open(str(audio_dest), "w") as f:
                f.setnchannels(2)
                f.setsampwidth(2)
                f.setframerate(48000)
                silent_frames = struct.pack("<h", 0) * int(48000 * min(duration, 2.0) * 2)
                f.writeframes(silent_frames)
            log_lines.append(f"[DRY RUN] Generated synthetic validation audio: {audio_dest.name}")
        except Exception:
            audio_dest.write_bytes(b"RIFF_SYNTHETIC_WAV_HEADER")

        elapsed_time = 0.05
        peak_alloc_gb = 0.0
        peak_res_gb = 0.0
        rtf = round(elapsed_time / max(duration, 0.1), 3)

        return {
            "status": "dry_run_success",
            "audio_file": audio_dest.name,
            "elapsed_seconds": elapsed_time,
            "peak_vram_allocated_gb": peak_alloc_gb,
            "peak_vram_reserved_gb": peak_res_gb,
            "real_time_factor": rtf,
            "simulated": True,
        }

    # Live neural generation
    import torch
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        instrumental=is_instrumental,
        vocal_language=vocal_language,
        bpm=bpm,
        duration=duration,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        shift=1.0,
        infer_method="ode",
        thinking=True,
        lm_temperature=0.85,
    )

    config = GenerationConfig(
        batch_size=1,
        use_random_seed=(seed < 0),
        seeds=[seed] if seed >= 0 else None,
        audio_format="flac",
    )

    print(f">>> [Seed {seed}] Generating (duration={duration}s, steps={inference_steps}, CFG={guidance_scale})...")
    result = generate_music(
        dit_handler,
        llm_handler,
        params,
        config,
        save_dir=str(candidate_dir),
    )

    elapsed_time = round(time.time() - start_time, 2)
    peak_alloc_gb = 0.0
    peak_res_gb = 0.0
    if torch.cuda.is_available():
        peak_alloc_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        peak_res_gb = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)

    rtf = round(elapsed_time / max(duration, 0.1), 3)

    if not result.success or not result.audios:
        err_msg = result.error or "generate_music returned failure without error message"
        log_lines.append(f"[ERROR] Candidate generation failed: {err_msg}")
        raise RuntimeError(f"ACE-Step generation failed: {err_msg}")

    src_audio = Path(result.audios[0]["path"])
    final_audio = candidate_dir / "audio.flac"
    if src_audio.resolve() != final_audio.resolve():
        shutil.copy2(src_audio, final_audio)

    # Validate output audio file existence and non-zero size
    if not final_audio.exists() or final_audio.stat().st_size < 1000:
        raise RuntimeError(f"Generated audio file {final_audio} is missing or empty.")

    return {
        "status": "success",
        "audio_file": final_audio.name,
        "elapsed_seconds": elapsed_time,
        "peak_vram_allocated_gb": peak_alloc_gb,
        "peak_vram_reserved_gb": peak_res_gb,
        "real_time_factor": rtf,
        "simulated": False,
    }


def write_manifest(batch_dir: Path, records: List[Dict[str, Any]], preset_name: str, mode: str) -> None:
    """Generate batch-level JSON and Markdown manifests."""
    manifest_data = {
        "batch_dir": str(batch_dir.resolve()),
        "preset": preset_name,
        "mode": mode,
        "total_candidates": len(records),
        "candidates": records,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(batch_dir / "batch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Markdown table manifest
    lines = [
        f"# Batch Manifest: {preset_name} [{mode}]",
        f"\n**Total Candidates:** {len(records)} | **Date:** {manifest_data['created_at']}\n",
        "| Candidate | Seed | Audio File | Wall Time | RTF | Peak VRAM | Status |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :--- |",
    ]
    for r in records:
        lines.append(
            f"| `{r['candidate_id']}` | `{r['seed']}` | `{r['audio_file']}` | {r['elapsed_seconds']}s | {r.get('real_time_factor', 'N/A')}x | {r['peak_vram_gb']} GB | {r['status']} |"
        )
    lines.append("")

    with open(batch_dir / "batch_manifest.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="ACE-Step 1.5 Lab Benchmark Generator")
    parser.add_argument("engine", nargs="?", default="ace-step", help="Model engine (default: ace-step)")
    parser.add_argument("preset", nargs="?", default=None, help="Preset name (reggaeton_nocturno, dembow_dominicano, reggaeton_2010s, smoke)")
    parser.add_argument("mode", nargs="?", default="instrumental", choices=["instrumental", "male-vocal"], help="Generation mode")
    parser.add_argument("--duration", type=float, default=None, help="Audio duration in seconds")
    parser.add_argument("--bpm", type=int, default=None, help="Explicit BPM")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds/candidates to generate in batch")
    parser.add_argument("--seed-list", type=int, nargs="+", default=None, help="Explicit list of seeds to run")
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG guidance scale (default: 7.0)")
    parser.add_argument("--steps", type=int, default=None, help="Inference denoising steps (default: 32)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit destination directory")
    parser.add_argument("--checkpoints-dir", type=Path, default=None, help="Path to checkpoints root")
    parser.add_argument("--presets-dir", type=Path, default=None, help="Directory containing preset JSON files")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"], help="Device to use")
    parser.add_argument("--smoke", action="store_true", help="Execute minimal live pipeline smoke test (5s duration, 8 steps)")
    parser.add_argument("--dry-run", action="store_true", help="Perform offline validation without loading neural models")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    presets_dir = args.presets_dir or (repo_root / "presets")
    checkpoints_dir = args.checkpoints_dir or Path(os.environ.get("CHECKPOINTS_DIR", repo_root / "checkpoints"))
    base_outputs_dir = repo_root / "outputs"
    lab_git_sha = get_lab_git_sha(repo_root)

    # 1. Resolve preset & smoke mode
    is_smoke = args.smoke or (args.engine == "smoke") or (args.preset == "smoke")
    preset_key = "smoke" if is_smoke else (args.preset or "reggaeton_nocturno")
    mode_key = args.mode

    preset_cfg = load_preset(preset_key, presets_dir)
    preset_name = preset_cfg.get("name", preset_key)
    mode_cfg = preset_cfg["modes"][mode_key]

    caption = mode_cfg["caption"]
    lyrics = mode_cfg["lyrics"]
    is_instrumental = mode_cfg.get("instrumental", mode_key == "instrumental")
    vocal_language = preset_cfg.get("vocal_language", "es")

    # Smoke test defaults to short 5s diagnostic
    if is_smoke:
        bpm = args.bpm if args.bpm is not None else 120
        duration = args.duration if args.duration is not None else 5.0
        seed = args.seed if args.seed is not None else 42
        guidance_scale = args.guidance_scale if args.guidance_scale is not None else 7.0
        inference_steps = args.steps if args.steps is not None else 8
        batch_seeds = [seed]
    else:
        bpm = args.bpm if args.bpm is not None else preset_cfg.get("bpm", 94)
        duration = args.duration if args.duration is not None else preset_cfg.get("default_duration", 30.0)
        seed = args.seed if args.seed is not None else preset_cfg.get("default_seed", 42)
        guidance_scale = args.guidance_scale if args.guidance_scale is not None else preset_cfg.get("default_guidance_scale", 7.0)
        inference_steps = args.steps if args.steps is not None else preset_cfg.get("default_inference_steps", 32)
        batch_seeds = build_batch_seeds(seed, args.seeds, args.seed_list)

    # 2. Setup Base Batch Directory
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = "smoke" if is_smoke else f"{preset_name}_{mode_key}"
    batch_dir = args.output_dir or (base_outputs_dir / f"{ts}_{prefix}")
    batch_dir.mkdir(parents=True, exist_ok=True)

    gpu_meta = get_gpu_metadata()

    print("=" * 75)
    if is_smoke:
        print(f"ACE-Step 1.5 Lab :: LIVE PIPELINE SMOKE TEST")
    else:
        print(f"ACE-Step 1.5 Lab :: Benchmark Generator ({preset_name} [{mode_key}])")
    print(f"Output Directory : {batch_dir.resolve()}")
    print(f"Candidates/Seeds : {len(batch_seeds)} {batch_seeds}")
    print(f"Duration: {duration}s | Steps: {inference_steps} | CFG: {guidance_scale} | BPM: {bpm}")
    print("=" * 75)

    # 3. Model Initialization (Loaded once in memory for all seeds)
    dit_handler = None
    llm_handler = None

    if not args.dry_run:
        resolved_device = "cuda" if (args.device == "cuda" or (args.device == "auto" and gpu_meta["cuda_available"])) else "cpu"
        dit_handler, llm_handler = initialize_ace_step_pipeline(checkpoints_dir, resolved_device)

    # 4. Sequential In-Memory Candidate Generation
    manifest_records = []
    overall_success = True

    for idx, cand_seed in enumerate(batch_seeds, start=1):
        cand_id = f"seed_{cand_seed}" if len(batch_seeds) > 1 else "output"
        cand_dir = batch_dir / cand_id if len(batch_seeds) > 1 else batch_dir
        cand_dir.mkdir(parents=True, exist_ok=True)

        cand_exec_id = str(uuid.uuid4())
        cand_start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_lines = [
            f"=== ACE-Step 1.5 Candidate Execution Log ===",
            f"Execution ID : {cand_exec_id}",
            f"Preset       : {preset_name} [{mode_key}]",
            f"Candidate ID : {cand_id} (Seed: {cand_seed})",
            f"Timestamp    : {cand_start_iso}",
        ]

        # Save prompt.txt
        prompt_file = cand_dir / "prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(f"# Preset: {preset_name} ({mode_key})\n")
            f.write(f"# Seed: {cand_seed} | BPM: {bpm} | Duration: {duration}s\n")
            f.write(f"# Language: {vocal_language}\n\n")
            f.write("# Caption\n")
            f.write(f"{caption}\n\n")
            f.write("# Lyric\n")
            f.write(f"{lyrics}\n")

        error_msg = None
        cand_res: Dict[str, Any] = {}

        try:
            cand_res = run_single_candidate(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                caption=caption,
                lyrics=lyrics,
                is_instrumental=is_instrumental,
                vocal_language=vocal_language,
                bpm=bpm,
                duration=duration,
                seed=cand_seed,
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                device="cuda" if (args.device == "cuda" or (args.device == "auto" and gpu_meta["cuda_available"])) else "cpu",
                candidate_dir=cand_dir,
                dry_run=args.dry_run,
                log_lines=log_lines,
            )
        except Exception as e:
            error_msg = str(e)
            overall_success = False
            log_lines.append(f"[FATAL ERROR] {e}")
            print(f"    [FAIL] Candidate {cand_id} failed: {e}", file=sys.stderr)

        cand_end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Build candidate metadata
        metadata = {
            "execution_id": cand_exec_id,
            "engine": "ace-step",
            "git_commit_sha": lab_git_sha,
            "upstream_commit_sha": PINNED_UPSTREAM_COMMIT,
            "models": {
                "diffusion_transformer": {
                    "repo_id": "ACE-Step/acestep-v15-xl-sft",
                    "revision": PINNED_MODEL_REVISIONS["acestep-v15-xl-sft"],
                },
                "language_model_planner": {
                    "repo_id": "ACE-Step/acestep-5Hz-lm-4B",
                    "revision": PINNED_MODEL_REVISIONS["acestep-5Hz-lm-4B"],
                },
                "audio_vae": {
                    "repo_id": "ACE-Step/Ace-Step1.5",
                    "subfolder": "vae",
                    "revision": PINNED_MODEL_REVISIONS["vae"],
                },
                "text_embedding": {
                    "repo_id": "ACE-Step/Ace-Step1.5",
                    "subfolder": "Qwen3-Embedding-0.6B",
                    "revision": PINNED_MODEL_REVISIONS["Qwen3-Embedding-0.6B"],
                },
            },
            "preset": {
                "name": preset_name,
                "mode": mode_key,
                "display_name": preset_cfg.get("display_name", preset_name),
            },
            "prompt": {
                "caption": caption,
                "lyrics": lyrics,
                "negative_prompt": "NO USER INPUT",
                "instrumental": is_instrumental,
                "vocal_language": vocal_language,
                "lm_planning_enabled": True,
            },
            "parameters": {
                "bpm": bpm,
                "duration_seconds": duration,
                "seed": cand_seed,
                "guidance_scale": guidance_scale,
                "inference_steps": inference_steps,
                "infer_method": "ode",
                "shift": 1.0,
            },
            "hardware": gpu_meta,
            "performance": {
                "elapsed_seconds": cand_res.get("elapsed_seconds", 0.0),
                "real_time_factor": cand_res.get("real_time_factor", 0.0),
                "peak_vram_allocated_gb": cand_res.get("peak_vram_allocated_gb", 0.0),
                "peak_vram_reserved_gb": cand_res.get("peak_vram_reserved_gb", 0.0),
            },
            "artifacts": {
                "audio_file": cand_res.get("audio_file", None),
                "prompt_file": prompt_file.name,
                "log_file": "run.log",
            },
            "timestamps": {
                "started_at": cand_start_iso,
                "completed_at": cand_end_iso,
            },
            "dry_run": args.dry_run,
            "smoke": is_smoke,
            "success": (error_msg is None),
            "error": error_msg,
        }

        with open(cand_dir / "generation_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(cand_dir / "run.log", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

        audio_full_path = cand_dir / cand_res.get("audio_file", "")
        try:
            audio_path_str = str(audio_full_path.relative_to(repo_root))
        except ValueError:
            audio_path_str = str(audio_full_path)

        manifest_records.append({
            "candidate_id": cand_id,
            "seed": cand_seed,
            "audio_file": cand_res.get("audio_file", "None"),
            "audio_path": audio_path_str,
            "elapsed_seconds": cand_res.get("elapsed_seconds", 0.0),
            "real_time_factor": cand_res.get("real_time_factor", 0.0),
            "peak_vram_gb": cand_res.get("peak_vram_allocated_gb", 0.0),
            "status": "PASS" if error_msg is None else "FAIL",
        })

    # 5. Write Batch Manifest
    if len(batch_seeds) > 1 or is_smoke:
        write_manifest(batch_dir, manifest_records, preset_name, mode_key)

    print("\n" + "=" * 75)
    if is_smoke:
        if overall_success:
            print(f"[SMOKE TEST PASSED] Real pipeline inference verified.")
            print(f"Artifacts: {batch_dir.resolve()}")
        else:
            print(f"[SMOKE TEST FAILED] Check logs in {batch_dir.resolve()}", file=sys.stderr)
            sys.exit(1)
    else:
        if overall_success:
            print(f"[SUCCESS] All {len(batch_seeds)} candidates generated.")
            print(f"Batch Manifest: {batch_dir / 'batch_manifest.md'}")
        else:
            print(f"[WARNING] Batch finished with errors. Inspect candidate logs.", file=sys.stderr)
            sys.exit(1)
    print("=" * 75)


if __name__ == "__main__":
    main()
