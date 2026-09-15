#!/usr/bin/env bash
# ==============================================================================
# run.sh - CLI Runner for ACE-Step 1.5 Evaluations & Smoke Testing
# Usage:
#   ./run.sh smoke                           # Real GPU live pipeline smoke test
#   ./run.sh ace-step <preset> <mode> [opts] # Single or batch generation
# Examples:
#   ./run.sh smoke
#   ./run.sh ace-step reggaeton_nocturno instrumental
#   ./run.sh ace-step dembow_dominicano instrumental --seeds 4
#   ./run.sh ace-step reggaeton_2010s male-vocal --duration 45 --seed 777
# ==============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Resolve python executable (prioritizing venv, then valid python3, then python)
PYTHON_EXEC=""
if [ -f "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_EXEC="${REPO_ROOT}/.venv/bin/python"
elif [ -f "${REPO_ROOT}/.venv/Scripts/python.exe" ]; then
    PYTHON_EXEC="${REPO_ROOT}/.venv/Scripts/python.exe"
elif command -v python3 &> /dev/null && python3 --version &> /dev/null; then
    PYTHON_EXEC="python3"
elif command -v python &> /dev/null && python --version &> /dev/null; then
    PYTHON_EXEC="python"
else
    PYTHON_EXEC="python3"
fi

show_help() {
    cat << EOF
music-gpu-lab - ACE-Step 1.5 Benchmark & Smoke Runner

Usage:
  ./run.sh smoke [options]
  ./run.sh ace-step <preset> <mode> [options]

Commands:
  smoke                 Short live pipeline probe (5s, 8 steps) verifying CUDA, weights, and WAV output

Presets:
  reggaeton_nocturno    Minimal nocturnal reggaeton around 94 BPM, dry muffled dembow, deep sub bass.
  dembow_dominicano     Raw modern Dominican dembow around 118 BPM, punchy snare, deep bassline.
  reggaeton_2010s       Early-2010s Puerto Rican reggaeton around 95 BPM, bright synth arpeggio motif.

Modes:
  instrumental          Beat/instrumental generation ([Instrumental])
  male-vocal            Spanish low male baritone vocal guide with structured verse/chorus

Options:
  --seeds <int>         Generate N independent candidates in a single in-memory session (default: 1)
  --seed <int>          Base seed for reproducibility (default: 42)
  --duration <seconds>  Audio generation duration in seconds (default: 30)
  --bpm <int>           Custom BPM (default: preset BPM)
  --guidance-scale <fl> CFG guidance scale for SFT model (default: 7.0)
  --steps <int>         Inference denoising steps (default: 32)
  --output-dir <path>   Explicit destination directory for outputs
  --dry-run             Simulate generation, validate prompts and record metadata without GPU
  --help                Show this help message

Examples:
  ./run.sh smoke
  ./run.sh ace-step reggaeton_nocturno instrumental
  ./run.sh ace-step dembow_dominicano instrumental --seeds 4
  ./run.sh ace-step reggaeton_nocturno male-vocal --duration 45 --seed 123
EOF
}

if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 1
fi

# Special command: smoke test
if [ "$1" = "smoke" ]; then
    shift
    exec "$PYTHON_EXEC" "${REPO_ROOT}/scripts/generate.py" --smoke "$@"
fi

# Standard preset execution
ENGINE="ace-step"
PRESET=""
MODE=""

if [ "$1" = "ace-step" ]; then
    if [ $# -lt 3 ]; then
        echo "Error: Missing preset and mode arguments."
        show_help
        exit 1
    fi
    ENGINE="$1"
    PRESET="$2"
    MODE="$3"
    shift 3
else
    if [ $# -lt 2 ]; then
        echo "Error: Missing mode argument."
        show_help
        exit 1
    fi
    PRESET="$1"
    MODE="$2"
    shift 2
fi

# Validate mode
if [ "$MODE" != "instrumental" ] && [ "$MODE" != "male-vocal" ]; then
    echo "Error: Invalid mode '$MODE'. Expected 'instrumental' or 'male-vocal'."
    exit 1
fi

# Execute Python runner forwarding any additional flags
exec "$PYTHON_EXEC" "${REPO_ROOT}/scripts/generate.py" "$ENGINE" "$PRESET" "$MODE" "$@"
