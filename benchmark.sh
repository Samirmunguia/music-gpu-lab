#!/usr/bin/env bash
# ==============================================================================
# benchmark.sh - First-Hour Automated Benchmark Suite for music-gpu-lab
# Evaluates: reggaeton_nocturno, dembow_dominicano, reggaeton_2010s
# Phase 1: Instrumental (30s clips, 4 seeds each = 12 tracks)
# Phase 2: Spanish Low Male Baritone Vocal (30s clips, 2 seeds each = 6 tracks)
# ==============================================================================

set -eo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RESET="\033[0m"

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

# Default parameters
DURATION=30
PHASE1_SEEDS=4
PHASE2_SEEDS=2
DRY_RUN_FLAG=""
OUTPUT_DIR=""

show_help() {
    cat << EOF
music-gpu-lab - First-Hour Benchmark Suite Runner

Usage:
  ./benchmark.sh [options]

Options:
  --duration <sec>       Clip duration in seconds (default: 30)
  --phase1-seeds <int>   Number of seeds per style for Phase 1 Instrumental (default: 4)
  --phase2-seeds <int>   Number of seeds per style for Phase 2 Male Vocal (default: 2)
  --output-dir <path>    Explicit output directory for benchmark
  --dry-run              Simulate entire benchmark suite without GPU
  --help                 Show this help message

Styles Evaluated:
  1. reggaeton_nocturno  (94 BPM)
  2. dembow_dominicano   (118 BPM)
  3. reggaeton_2010s     (95 BPM)

Default Total: 18 diagnostic tracks (12 instrumental + 6 vocal)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --phase1-seeds)
            PHASE1_SEEDS="$2"
            shift 2
            ;;
        --phase2-seeds)
            PHASE2_SEEDS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN_FLAG="--dry-run"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

TS=$(date -u +"%Y%m%d_%H%M%S")
BENCHMARK_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/benchmark_${TS}}"
mkdir -p "$BENCHMARK_DIR"

echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}     music-gpu-lab :: First-Hour Rented-GPU Benchmark Suite     ${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo -e "Benchmark Destination : ${BENCHMARK_DIR}"
echo -e "Clip Duration         : ${DURATION}s"
echo -e "Phase 1 Seeds (Inst)  : ${PHASE1_SEEDS} per style (Total: $((PHASE1_SEEDS * 3)))"
echo -e "Phase 2 Seeds (Vocal) : ${PHASE2_SEEDS} per style (Total: $((PHASE2_SEEDS * 3)))"
echo -e "Total Target Tracks   : $(((PHASE1_SEEDS * 3) + (PHASE2_SEEDS * 3)))"
echo -e "${BOLD}================================================================${RESET}\n"

STYLES=("reggaeton_nocturno" "dembow_dominicano" "reggaeton_2010s")

# ------------------------------------------------------------------------------
# Phase 1: Instrumental Tracks (4 seeds per style)
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}>>> STARTING PHASE 1: INSTRUMENTAL BENCHMARK (${PHASE1_SEEDS} seeds/style)${RESET}"
for style in "${STYLES[@]}"; do
    echo -e "\n${BOLD}[Phase 1] Style: ${style} [instrumental]${RESET}"
    STYLE_OUT="${BENCHMARK_DIR}/${style}_instrumental"
    # shellcheck disable=SC2086
    "$PYTHON_EXEC" "${REPO_ROOT}/scripts/generate.py" ace-step "$style" instrumental \
        --duration "$DURATION" \
        --seeds "$PHASE1_SEEDS" \
        --output-dir "$STYLE_OUT" \
        $DRY_RUN_FLAG
done

# ------------------------------------------------------------------------------
# Phase 2: Guide Vocal Tracks (2 seeds per style)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${CYAN}>>> STARTING PHASE 2: SPANISH LOW MALE BARITONE VOCAL (${PHASE2_SEEDS} seeds/style)${RESET}"
for style in "${STYLES[@]}"; do
    echo -e "\n${BOLD}[Phase 2] Style: ${style} [male-vocal]${RESET}"
    STYLE_OUT="${BENCHMARK_DIR}/${style}_male-vocal"
    # shellcheck disable=SC2086
    "$PYTHON_EXEC" "${REPO_ROOT}/scripts/generate.py" ace-step "$style" male-vocal \
        --duration "$DURATION" \
        --seeds "$PHASE2_SEEDS" \
        --output-dir "$STYLE_OUT" \
        $DRY_RUN_FLAG
done

# ------------------------------------------------------------------------------
# Compile Master Index
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Compiling Master Benchmark Index]${RESET}"
"$PYTHON_EXEC" "${REPO_ROOT}/scripts/index_benchmark.py" "$BENCHMARK_DIR"

echo -e "\n${BOLD}================================================================${RESET}"
echo -e "${GREEN}${BOLD}             BENCHMARK SUITE EXECUTION COMPLETE                 ${RESET}"
echo -e "${BOLD}================================================================${RESET}\n"

if [ -f "${BENCHMARK_DIR}/benchmark_index.md" ]; then
    cat "${BENCHMARK_DIR}/benchmark_index.md"
fi
echo -e "${BOLD}================================================================${RESET}"
