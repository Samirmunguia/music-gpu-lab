# music-gpu-lab

A minimal, hardened evaluation laboratory for **ACE-Step 1.5** on disposable rented NVIDIA GPU instances:
* **A100 80GB / H200 (80–141 GB VRAM)**
* **RTX PRO 6000 Blackwell (96 GB VRAM)**
* **RTX 6000 Ada (48 GB VRAM)**

*(Note on GPU naming: RTX 6000 Ada carries 48 GB VRAM, whereas RTX PRO 6000 Blackwell carries 96 GB VRAM).*

Optimized for **first-hour operational readiness**: deterministic batching, in-memory model reuse, pinned upstream revisions, automated smoke testing, and benchmark indexing.

### Pinned Model & Upstream Architecture
* **Upstream ACE-Step Repository:** `https://github.com/ace-step/ACE-Step-1.5.git` @ commit `ca1e85fe9430179831e6bc6be790c332190a3866`
* **Diffusion Model (DiT):** `ACE-Step/acestep-v15-xl-sft` @ rev `d06de46b4622f781cf07f4a013a67d591ca52819` (4B DiT SFT with CFG)
* **Language Model Planner (LM):** `ACE-Step/acestep-5Hz-lm-4B` @ rev `0a3ec94b557aea7d508da38b31cfe7341f6ff737` (4B Qwen3 music planner)
* **Shared Audio VAE:** `ACE-Step/Ace-Step1.5` (`vae`) @ rev `19671f406d603126926c1b7e2adc169acbcade22`
* **Shared Text Embedding:** `ACE-Step/Ace-Step1.5` (`Qwen3-Embedding-0.6B`) @ rev `19671f406d603126926c1b7e2adc169acbcade22`

Zero cloud-provider lock-in. No daemons. No background services. No web UI overhead.

---

## One-Command Rented-GPU Quickstart

On a fresh rented GPU instance, run:

```bash
# 1. Clone laboratory repository
git clone https://github.com/<your-org>/music-gpu-lab.git
cd music-gpu-lab

# 2. One-command preparation (audits hardware, bootstraps venv & weights, runs real live smoke test)
./prepare_gpu.sh

# 3. Execute first-hour benchmark suite (18 diagnostic tracks across 3 styles)
./benchmark.sh

# 4. Inspect consolidated benchmark table
cat outputs/benchmark_*/benchmark_index.md

# 5. Retrieve outputs and destroy GPU instance
tar -czvf results.tar.gz outputs/
```

---

## Individual Step-by-Step Workflow

If you prefer running individual stages manually:

### 1. Pre-Flight Hardware & Storage Audit
```bash
./verify_gpu.sh
```
Validates NVIDIA driver, exact GPU model, total/free VRAM, CUDA visibility, system RAM (≥32GB recommended), disk space (≥40GB required for 4B models), and connectivity to HuggingFace.

### 2. Bootstrap Dependencies & Model Weights
```bash
./bootstrap.sh
```
Safe to rerun. Creates `.venv`, installs locked Python dependencies, pins upstream ACE-Step 1.5, downloads pinned weights (~29.87 GB), and skips already cached files.

### 3. Real GPU Live Smoke Test
```bash
./run.sh smoke
```
Executes a minimal 5-second, 8-step live inference on the actual XL-SFT + LM-4B pipeline. Verifies CUDA kernel execution, model initialization, and valid audio rendering before launching long benchmarks.

---

## Benchmark Presets

| Preset | BPM | Description |
| :--- | :---: | :--- |
| `reggaeton_nocturno` | 94 | Minimal nocturnal reggaeton, dry/muffled dembow groove, deep 808 sub bass, sparse hypnotic synth pluck motif. |
| `dembow_dominicano` | 118 | Raw modern Dominican dembow, minimal hard percussion, punchy modern snare, heavy sub bass, street cadence. |
| `reggaeton_2010s` | 95 | Early-2010s Puerto Rican reggaeton character, dirty dry dembow beat, memorable bright digital synth arpeggio lead motif. |

Each preset supports:
* `instrumental`: Beat-focused arrangement (`[Instrumental]`).
* `male-vocal`: Spanish low male baritone guide vocal with structured verse and chorus cues.

---

## Batch Generation (In-Memory Model Reuse)

To avoid reloading 4B weights into VRAM on every candidate (~2–3 minutes overhead saved per track), batch mode loads the model once and evaluates multiple seeds sequentially:

```bash
# Generate 4 distinct instrumental candidates
./run.sh ace-step dembow_dominicano instrumental --seeds 4

# Generate 2 vocal candidates with 45s duration
./run.sh ace-step reggaeton_nocturno male-vocal --seeds 2 --duration 45
```

Outputs are structured with individual candidate subdirectories and a unified batch manifest:
```text
outputs/20260915_160000_dembow_dominicano_instrumental/
├── seed_42/
│   ├── audio.flac
│   ├── prompt.txt
│   ├── generation_metadata.json
│   └── run.log
├── seed_43/
│   └── ...
├── batch_manifest.json
└── batch_manifest.md
```

---

## First-Hour Benchmark Suite (`benchmark.sh`)

Automates a standardized evaluation matrix designed to complete within ~20–35 minutes on an A100 80GB / H200:

* **Phase 1 (Instrumental):** 3 styles × 4 seeds = **12 tracks** (30s clips)
* **Phase 2 (Male Guide Vocal):** 3 styles × 2 seeds = **6 tracks** (30s clips)
* **Total:** **18 diagnostic tracks**

```bash
# Run default 18-track benchmark suite
./benchmark.sh

# Custom duration and seed count
./benchmark.sh --duration 35 --phase1-seeds 2 --phase2-seeds 1

# Offline simulation (no GPU required)
./benchmark.sh --dry-run
```

### Benchmark Summary Output
Automatically compiled into:
* `outputs/benchmark_<timestamp>/benchmark_index.md` (Markdown table)
* `outputs/benchmark_<timestamp>/benchmark_index.csv` (CSV spreadsheet)
* `outputs/benchmark_<timestamp>/benchmark_index.json` (Machine-readable)

Example Index Table:
| Style / Preset | Mode | Seed | Dur (s) | BPM | CFG | Wall Time | RTF | Peak VRAM | Audio File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `reggaeton_nocturno` | `instrumental` | `42` | 30.0s | 94 | 7.0 | 28.4s | 0.95x | 21.4 GB | `seed_42/audio.flac` |
| `reggaeton_nocturno` | `instrumental` | `43` | 30.0s | 94 | 7.0 | 27.8s | 0.93x | 21.4 GB | `seed_43/audio.flac` |

---

## Telemetry Captured per Candidate

Every `generation_metadata.json` records:
* **Execution & Commits:** `execution_id`, `git_commit_sha` (lab repo), `upstream_commit_sha` (ACE-Step upstream).
* **Models & Revisions:** Pinned HuggingFace commit SHAs for DiT, LM, VAE, and text embedding.
* **Prompt & Conditioning:** Complete caption, lyrics, negative prompt (`"NO USER INPUT"`), vocal language, `lm_planning_enabled`.
* **Audio Parameters:** `bpm`, `duration_seconds`, `seed`, `guidance_scale`, `inference_steps`, `infer_method`, `shift`.
* **Hardware:** GPU device name, total VRAM, driver version, CUDA runtime version.
* **Performance Telemetry:** `elapsed_seconds`, `real_time_factor` (elapsed / duration), `peak_vram_allocated_gb`, `peak_vram_reserved_gb`.

---

## Storage & Persistent Volumes

For cloud providers offering persistent network storage volumes:

```bash
export CHECKPOINTS_DIR="/mnt/volume/checkpoints"
export HF_HOME="/mnt/volume/.cache/huggingface"

./prepare_gpu.sh
```
If checkpoints already exist in `$CHECKPOINTS_DIR`, `bootstrap.sh` validates file integrity in ~2 seconds without re-downloading.

---

## Offline Validation (Local Machine, No GPU Required)

Run the automated test suite locally to verify schemas, argument parsing, smoke test structure, and batching logic:

```bash
python scripts/test_dry_run.py
```
