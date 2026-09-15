#!/usr/bin/env python3
"""
Diagnostic Ablation Runner & Self-Analysis for Dominican Dembow (music-gpu-lab).
Executes high-information ablations (A1-A9) testing LM prompt rewrites, CFG,
sampling temperature, inference steps, thinking, and negative prompts on an A100 GPU.
Includes ACE-Step understand_music self-analysis.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", REPO_ROOT / "checkpoints"))
PRESETS_DIR = REPO_ROOT / "presets"
DIAGNOSTIC_OUTPUT_DIR = REPO_ROOT / "outputs" / "diagnostic_round1_dembow"

# Ensure acestep is importable
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
    """Recursively convert object to JSON-serializable types, omitting tensors/arrays."""
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


def get_baseline_control_info() -> Dict[str, Any]:
    """Retrieve metadata and audio codes for existing baseline dembow seed 42."""
    baseline_dir = REPO_ROOT / "outputs" / "benchmark_20260915_215146" / "dembow_dominicano_instrumental" / "seed_42"
    audio_path = baseline_dir / "audio.flac"
    meta_path = baseline_dir / "generation_metadata.json"
    
    codes_str = ""
    log_path = REPO_ROOT / "benchmark.log"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if "Debug output text: <|audio_code_" in line and idx < 750:
                parts = line.split("Debug output text: ")
                if len(parts) > 1:
                    codes_str = parts[1].strip()
                    break

    meta_data = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

    return {
        "variant_id": "CONTROL",
        "name": "baseline_control",
        "description": "Existing baseline dembow seed 42 (untouched control)",
        "audio_path": str(audio_path.resolve()) if audio_path.exists() else None,
        "audio_codes": codes_str,
        "metadata": meta_data,
        "cot_caption": (
            "A quirky and hypnotic instrumental track built around a repetitive, catchy "
            "melodica hook that sounds like a toy keyboard melody. A dry, punchy dembow-style "
            "drum machine beat provides a steady, danceable groove, anchored by a deep sub-bass "
            "synth. The arrangement is minimalist and loop-based, evolving through the subtle "
            "addition and subtraction of rhythmic layers, including shaker-like percussion and "
            "brief structural breaks where the beat drops out before slamming back in. The overall "
            "vibe is playful, groovy, and slightly lo-fi."
        ),
        "cot_bpm": 118,
        "cot_keyscale": "B♭ minor",
    }


def define_ablation_variants() -> List[Dict[str, Any]]:
    default_caption = (
        "raw modern Dominican dembow, 118 bpm, minimal hard-hitting percussion, "
        "characteristic punchy modern snare, syncopated dembow rhythm, deep distorted 808 bassline, "
        "raw urban street energy, stripped back arrangement, instrumental"
    )
    default_lyrics = "[Instrumental]"
    
    return [
        {
            "id": "A1",
            "name": "cot_caption_off",
            "description": "Disables LM CoT caption rewriting (feeds raw human prompt to DiT)",
            "params_override": {
                "use_cot_caption": False,
            },
        },
        {
            "id": "A2",
            "name": "thinking_off",
            "description": "Disables LM internal thinking entirely (direct metadata inference)",
            "params_override": {
                "thinking": False,
            },
        },
        {
            "id": "A3",
            "name": "lm_temperature_low",
            "description": "Low LM temperature (0.55) to reduce hallucinatory genre drift",
            "params_override": {
                "lm_temperature": 0.55,
            },
        },
        {
            "id": "A4",
            "name": "dit_cfg_low",
            "description": "Lower DiT CFG guidance scale (5.0 vs 7.0) for relaxed adherence",
            "params_override": {
                "guidance_scale": 5.0,
            },
        },
        {
            "id": "A5",
            "name": "dit_cfg_high",
            "description": "Higher DiT CFG guidance scale (9.0 vs 7.0) for strict prompt adherence",
            "params_override": {
                "guidance_scale": 9.0,
            },
        },
        {
            "id": "A6",
            "name": "steps_64",
            "description": "Increased diffusion steps (64 vs 32) to test if acoustic clarity improves",
            "params_override": {
                "inference_steps": 64,
            },
        },
        {
            "id": "A7",
            "name": "lm_cfg_high",
            "description": "High LM CFG scale (3.0 vs 1.0) during planning token generation",
            "params_override": {
                "lm_cfg_scale": 3.0,
            },
        },
        {
            "id": "A8",
            "name": "targeted_negative",
            "description": "Explicit negative prompt steering LM away from generic pop / gloss",
            "params_override": {
                "lm_negative_prompt": "generic Latin pop, glossy pop-reggaeton, EDM, fast generic dembow",
            },
        },
        {
            "id": "A9",
            "name": "short_direct_prompt",
            "description": "Short direct prompt with use_cot_caption=False (tests direct tag adherence)",
            "params_override": {
                "caption": "raw Dominican dembow, 118 BPM, minimal percussion, hard modern snare, deep bass",
                "use_cot_caption": False,
            },
        },
    ]


def run_diagnostics(device: str = "cuda"):
    print("=" * 80)
    print("   music-gpu-lab :: DIAGNOSTIC ABLATION ROUND 1 (Dominican Dembow)   ")
    print("=" * 80)
    DIAGNOSTIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    gpu_meta = get_gpu_metadata()
    print(f"Hardware: {gpu_meta['device_name']} ({gpu_meta['total_vram_gb']} GB VRAM)")
    print(f"Output Directory: {DIAGNOSTIC_OUTPUT_DIR.resolve()}\n")

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

    base_caption = (
        "raw modern Dominican dembow, 118 bpm, minimal hard-hitting percussion, "
        "characteristic punchy modern snare, syncopated dembow rhythm, deep distorted 808 bassline, "
        "raw urban street energy, stripped back arrangement, instrumental"
    )
    base_lyrics = "[Instrumental]"
    base_bpm = 118
    base_duration = 30.0
    base_seed = 42

    control_info = get_baseline_control_info()
    variants = define_ablation_variants()
    
    variant_results: List[Dict[str, Any]] = []

    for var in variants:
        var_id = var["id"]
        var_name = var["name"]
        var_desc = var["description"]
        var_dir = DIAGNOSTIC_OUTPUT_DIR / f"{var_id}_{var_name}"
        var_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 80)
        print(f">>> Executing Variant [{var_id}] {var_name}")
        print(f"    Description: {var_desc}")
        print(f"    Overrides  : {var['params_override']}")
        print("-" * 80)

        p_caption = var["params_override"].get("caption", base_caption)
        p_lyrics = var["params_override"].get("lyrics", base_lyrics)
        p_bpm = var["params_override"].get("bpm", base_bpm)
        p_duration = var["params_override"].get("duration", base_duration)
        p_seed = var["params_override"].get("seed", base_seed)
        p_thinking = var["params_override"].get("thinking", True)
        p_use_cot_caption = var["params_override"].get("use_cot_caption", True)
        p_use_cot_metas = var["params_override"].get("use_cot_metas", True)
        p_lm_temp = var["params_override"].get("lm_temperature", 0.85)
        p_lm_cfg = var["params_override"].get("lm_cfg_scale", 1.0)
        p_lm_neg = var["params_override"].get("lm_negative_prompt", "")
        p_guidance = var["params_override"].get("guidance_scale", 7.0)
        p_steps = var["params_override"].get("inference_steps", 32)
        p_infer_method = var["params_override"].get("infer_method", "ode")
        p_shift = var["params_override"].get("shift", 1.0)

        params = GenerationParams(
            task_type="text2music",
            caption=p_caption,
            lyrics=p_lyrics,
            instrumental=True,
            vocal_language="es",
            bpm=p_bpm,
            duration=p_duration,
            inference_steps=p_steps,
            guidance_scale=p_guidance,
            seed=p_seed,
            shift=p_shift,
            infer_method=p_infer_method,
            thinking=p_thinking,
            use_cot_caption=p_use_cot_caption,
            use_cot_metas=p_use_cot_metas,
            lm_temperature=p_lm_temp,
            lm_cfg_scale=p_lm_cfg,
            lm_negative_prompt=p_lm_neg if p_lm_neg else None,
        )

        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[p_seed],
            audio_format="flac",
        )

        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        error_msg = None
        gen_res = None
        cand_meta: Dict[str, Any] = {}
        audio_codes_str = ""

        try:
            gen_res = generate_music(
                dit_handler,
                llm_handler,
                params,
                config,
                save_dir=str(var_dir),
            )
            elapsed = round(time.time() - t0, 2)
            peak_alloc = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
            peak_res = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
            rtf = round(elapsed / max(p_duration, 0.1), 3)

            if not gen_res.success or not gen_res.audios:
                raise RuntimeError(gen_res.error or "generate_music reported failure")

            src_audio = Path(gen_res.audios[0]["path"])
            dest_audio = var_dir / "audio.flac"
            if src_audio.resolve() != dest_audio.resolve():
                shutil.copy2(src_audio, dest_audio)

            raw_codes = gen_res.audios[0].get("params", {}).get("audio_codes")
            if raw_codes and isinstance(raw_codes, str) and raw_codes.strip():
                audio_codes_str = raw_codes.strip()
                (var_dir / "audio_codes.txt").write_text(audio_codes_str, encoding="utf-8")

            extra = gen_res.extra_outputs or {}
            lm_meta = extra.get("lm_metadata") or {}
            time_costs = extra.get("time_costs") or {}

            cand_meta = {
                "variant_id": var_id,
                "variant_name": var_name,
                "description": var_desc,
                "execution_id": str(uuid.uuid4()),
                "timestamps": {
                    "started_at": start_iso,
                    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
                "inputs": {
                    "caption": p_caption,
                    "lyrics": p_lyrics,
                    "bpm": p_bpm,
                    "duration_seconds": p_duration,
                    "seed": p_seed,
                },
                "configuration": {
                    "thinking": p_thinking,
                    "use_cot_caption": p_use_cot_caption,
                    "use_cot_metas": p_use_cot_metas,
                    "lm_temperature": p_lm_temp,
                    "lm_cfg_scale": p_lm_cfg,
                    "lm_negative_prompt": p_lm_neg,
                    "guidance_scale": p_guidance,
                    "inference_steps": p_steps,
                    "infer_method": p_infer_method,
                    "shift": p_shift,
                },
                "lm_telemetry": {
                    "cot_caption": lm_meta.get("caption", None),
                    "cot_bpm": lm_meta.get("bpm", None),
                    "cot_keyscale": lm_meta.get("keyscale", None),
                    "cot_timesignature": lm_meta.get("timesignature", None),
                    "cot_vocal_language": lm_meta.get("vocal_language", None),
                    "raw_lm_metadata": make_json_safe(lm_meta),
                    "time_costs": make_json_safe(time_costs),
                    "has_audio_codes": bool(audio_codes_str),
                },
                "performance": {
                    "elapsed_seconds": elapsed,
                    "real_time_factor": rtf,
                    "peak_vram_allocated_gb": peak_alloc,
                    "peak_vram_reserved_gb": peak_res,
                },
                "hardware": gpu_meta,
                "artifacts": {
                    "audio_file": "audio.flac",
                    "audio_codes_file": "audio_codes.txt" if audio_codes_str else None,
                },
                "success": True,
                "error": None,
            }

            print(f"    [OK] Elapsed: {elapsed}s | RTF: {rtf}x | Peak VRAM: {peak_alloc} GB")
            if lm_meta.get("caption"):
                print(f"    [LM CoT Caption]: {lm_meta.get('caption')[:120]}...")

        except Exception as e:
            error_msg = str(e)
            print(f"    [FAIL] Variant {var_id} error: {e}", file=sys.stderr)
            cand_meta = {
                "variant_id": var_id,
                "variant_name": var_name,
                "description": var_desc,
                "success": False,
                "error": error_msg,
            }

        with open(var_dir / "diagnostic_metadata.json", "w", encoding="utf-8") as f:
            json.dump(cand_meta, f, indent=2)

        with open(var_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(f"# Variant: {var_id} - {var_name}\n")
            f.write(f"# Description: {var_desc}\n")
            f.write(f"# Overrides: {json.dumps(var['params_override'])}\n\n")
            f.write(f"# Caption\n{p_caption}\n\n# Lyric\n{p_lyrics}\n")

        var["result_meta"] = cand_meta
        var["audio_codes"] = audio_codes_str
        var["audio_path"] = str((var_dir / "audio.flac").resolve())
        variant_results.append(var)

    # 3. SELF-ANALYSIS: Run understand_music on [CONTROL, A1, A2, A9]
    print("\n" + "=" * 80)
    print(">>> EXECUTING SELF-ANALYSIS (understand_music)")
    print("=" * 80)

    self_analysis_targets = [
        ("CONTROL", "Baseline Dembow (Seed 42)", control_info["audio_codes"]),
    ]
    for v in variant_results:
        if v["id"] in ["A1", "A2", "A9"]:
            self_analysis_targets.append((v["id"], v["name"], v["audio_codes"]))

    self_analysis_records = []
    for tgt_id, tgt_name, tgt_codes in self_analysis_targets:
        print(f"\n>>> Running understand_music for [{tgt_id}] {tgt_name}...")
        if not tgt_codes:
            print(f"    [SKIP/NOTE] No audio semantic codes available for {tgt_id} (e.g. thinking=False).")
            record = {
                "target_id": tgt_id,
                "target_name": tgt_name,
                "success": False,
                "note": "No audio semantic codes generated (thinking=False or direct DiT)",
                "caption": "N/A",
                "bpm": None,
                "key": None,
                "language": None,
            }
        else:
            try:
                und_res = understand_music(
                    llm_handler=llm_handler,
                    audio_codes=tgt_codes,
                    temperature=0.85,
                )
                record = {
                    "target_id": tgt_id,
                    "target_name": tgt_name,
                    "success": und_res.success,
                    "caption": und_res.caption,
                    "bpm": und_res.bpm,
                    "key": und_res.keyscale,
                    "language": und_res.language,
                    "lyrics": und_res.lyrics,
                    "status_message": und_res.status_message,
                    "error": und_res.error,
                }
                print(f"    [Understand Result] Success: {und_res.success}")
                print(f"    BPM: {und_res.bpm} | Key: {und_res.keyscale} | Lang: {und_res.language}")
                print(f"    Understood Caption: {und_res.caption}")
            except Exception as e:
                print(f"    [ERROR] understand_music failed: {e}")
                record = {
                    "target_id": tgt_id,
                    "target_name": tgt_name,
                    "success": False,
                    "error": str(e),
                }

        self_analysis_records.append(record)
        if tgt_id != "CONTROL":
            cand_folder = DIAGNOSTIC_OUTPUT_DIR / f"{tgt_id}_{tgt_name}"
            with open(cand_folder / "self_analysis.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

    with open(DIAGNOSTIC_OUTPUT_DIR / "self_analysis.json", "w", encoding="utf-8") as f:
        json.dump(self_analysis_records, f, indent=2)

    # 4. Generate Master Diagnostic Index & Summary Table
    print("\n" + "=" * 80)
    print(">>> COMPILING DIAGNOSTIC MANIFEST AND COMPARISON TABLE")
    print("=" * 80)

    all_summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": gpu_meta,
        "control": control_info,
        "variants": [v["result_meta"] for v in variant_results],
        "self_analysis": self_analysis_records,
    }
    with open(DIAGNOSTIC_OUTPUT_DIR / "diagnostic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)

    md_lines = [
        "# Dominican Dembow Diagnostic Ablation Report (A1–A9)",
        f"\n**Execution Date:** {all_summary['timestamp']} | **GPU:** {gpu_meta['device_name']} (80 GB)\n",
        "## 1. Ablation Overview & Technical Metrics\n",
        "| ID | Variant Name | Core Modification | Wall Time | RTF | Peak VRAM | Status |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
        f"| `CTRL` | `baseline_control` | Standard benchmark baseline | 16.03s | 0.534x | 19.95 GB | PASS |",
    ]
    for v in variant_results:
        rm = v.get("result_meta", {})
        perf = rm.get("performance", {})
        md_lines.append(
            f"| `{v['id']}` | `{v['name']}` | {v['description']} | "
            f"{perf.get('elapsed_seconds', 'N/A')}s | {perf.get('real_time_factor', 'N/A')}x | "
            f"{perf.get('peak_vram_allocated_gb', 'N/A')} GB | {'PASS' if rm.get('success') else 'FAIL'} |"
        )

    md_lines.append("\n## 2. Prompt Rewriting Comparison (Human Input vs 4B LM CoT vs DiT Input)\n")
    md_lines.append(f"**Original Human Caption:**\n> *\"{base_caption}\"*\n")
    md_lines.append(f"**Baseline LM CoT Rewritten Caption:**\n> *\"{control_info['cot_caption']}\"*\n")

    for v in variant_results:
        rm = v.get("result_meta", {})
        cfg = rm.get("configuration", {})
        lm_tele = rm.get("lm_telemetry", {})
        cot_cap = lm_tele.get("cot_caption") or "(None / Thinking Off)"
        dit_actual_prompt = (
            cot_cap if cfg.get("use_cot_caption", True) and cot_cap != "(None / Thinking Off)"
            else rm.get("inputs", {}).get("caption")
        )
        md_lines.append(f"### Variant `{v['id']}`: `{v['name']}`")
        md_lines.append(f"- **Overrides**: `{json.dumps(v['params_override'])}`")
        md_lines.append(f"- **DiT Received Prompt**: *\"{dit_actual_prompt}\"*")
        if cot_cap != "(None / Thinking Off)":
            md_lines.append(f"- **LM CoT Generated Caption**: *\"{cot_cap}\"*")
        md_lines.append("")

    md_lines.append("## 3. ACE-Step Self-Analysis (`understand_music`)\n")
    md_lines.append("| Target ID | Variant Name | Understood Caption | Understood BPM | Understood Key | Understood Lang |")
    md_lines.append("| :--- | :--- | :--- | :---: | :---: | :---: |")
    for sa in self_analysis_records:
        cap_snip = sa.get("caption", "N/A")
        if len(cap_snip) > 90:
            cap_snip = cap_snip[:87] + "..."
        md_lines.append(
            f"| `{sa['target_id']}` | `{sa['target_name']}` | {cap_snip} | {sa.get('bpm', 'N/A')} | {sa.get('key', 'N/A')} | {sa.get('language', 'N/A')} |"
        )
    md_lines.append("")

    summary_md_path = DIAGNOSTIC_OUTPUT_DIR / "diagnostic_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n[OK] Diagnostic summary written to: {summary_md_path.resolve()}")
    print("=" * 80)
    print("DIAGNOSTIC ROUND 1 EXECUTION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostics()
