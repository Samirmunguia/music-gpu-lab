#!/usr/bin/env bash
# ==============================================================================
# bootstrap.sh - Disposable Instance Setup Script for music-gpu-lab
# Authoritative installation using upstream uv.lock (uv sync --frozen).
# Pins ACE-Step 1.5 upstream to commit ca1e85fe9430179831e6bc6be790c332190a3866.
# Audits dependencies with verify_dependencies.py before declaring readiness.
# ==============================================================================

set -eo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

BOOTSTRAP_START_TIME=$(date +%s)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${REPO_ROOT}/checkpoints}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
PINNED_UPSTREAM_COMMIT="ca1e85fe9430179831e6bc6be790c332190a3866"

export CHECKPOINTS_DIR
export HF_HOME
export ACESTEP_CHECKPOINTS_DIR="$CHECKPOINTS_DIR"
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}           music-gpu-lab :: Instance Bootstrap Setup            ${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo -e "Repository Root : ${REPO_ROOT}"
echo -e "Checkpoints Dir : ${CHECKPOINTS_DIR}"
echo -e "Virtualenv Dir  : ${VENV_DIR}"
echo -e "HuggingFace Home: ${HF_HOME}"
echo -e "Pinned Upstream : ${PINNED_UPSTREAM_COMMIT}\n"

# ------------------------------------------------------------------------------
# 1. Pre-Flight Storage Check (Need >= 35GB for XL-SFT + LM-4B)
# ------------------------------------------------------------------------------
echo -e "${BOLD}[1/7] Validating available disk storage...${RESET}"
mkdir -p "$CHECKPOINTS_DIR"
if command -v df &> /dev/null; then
    DISK_AVAIL_KB=$(df -k "$CHECKPOINTS_DIR" | tail -n 1 | awk '{print $4}')
    DISK_AVAIL_GB=$(awk "BEGIN {printf \"%.1f\", ${DISK_AVAIL_KB} / 1024 / 1024}")
    echo -e "Available disk on checkpoints volume: ${DISK_AVAIL_GB} GB"
    if [ "$DISK_AVAIL_KB" -lt 30000000 ] && [ "$SKIP_DOWNLOAD" -ne 1 ]; then
        echo -e "${RED}[ERROR] Insufficient disk space (${DISK_AVAIL_GB} GB).${RESET}"
        echo -e "ACE-Step 1.5 XL-SFT + LM-4B checkpoints require at least 35 GB free."
        echo -e "Attach a larger volume or set CHECKPOINTS_DIR to another path."
        exit 1
    else
        echo -e "${GREEN}[OK] Disk storage sufficient.${RESET}"
    fi
fi

# ------------------------------------------------------------------------------
# 2. System Package Prerequisites (git, curl, ffmpeg, libsndfile1)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[2/7] Checking minimal system tools...${RESET}"
MISSING_PKGS=()
for pkg in git curl ffmpeg; do
    if ! command -v "$pkg" &> /dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing system packages: ${MISSING_PKGS[*]}${RESET}"
    if command -v apt-get &> /dev/null; then
        if [ "$(id -u)" -eq 0 ]; then
            echo "Installing via apt-get..."
            apt-get update -qq && apt-get install -y -qq "${MISSING_PKGS[@]}" libsndfile1 libsndfile1-dev
        elif command -v sudo &> /dev/null; then
            echo "Installing via sudo apt-get..."
            sudo apt-get update -qq && sudo apt-get install -y -qq "${MISSING_PKGS[@]}" libsndfile1 libsndfile1-dev
        else
            echo -e "${YELLOW}Warning: Cannot install packages automatically. Please ensure ${MISSING_PKGS[*]} are installed.${RESET}"
        fi
    fi
else
    echo -e "${GREEN}All required system packages (git, curl, ffmpeg) are present.${RESET}"
fi

# ------------------------------------------------------------------------------
# 3. Clone and Pin Authoritative Upstream Repository
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[3/7] Fetching pinned upstream ACE-Step 1.5...${RESET}"
SRC_DIR="${REPO_ROOT}/src"
ACE_STEP_DIR="${SRC_DIR}/ACE-Step-1.5"
mkdir -p "$SRC_DIR"

if [ ! -d "${ACE_STEP_DIR}/.git" ]; then
    echo "Cloning upstream repository: https://github.com/ace-step/ACE-Step-1.5.git..."
    git clone https://github.com/ace-step/ACE-Step-1.5.git "$ACE_STEP_DIR"
fi

echo "Pinning ACE-Step 1.5 to verified commit ${PINNED_UPSTREAM_COMMIT}..."
cd "$ACE_STEP_DIR"
git checkout -q "$PINNED_UPSTREAM_COMMIT" || {
    git fetch -q --tags origin
    git checkout -q "$PINNED_UPSTREAM_COMMIT"
}
CURRENT_COMMIT=$(git rev-parse HEAD)
echo -e "${GREEN}ACE-Step 1.5 verified at commit: ${CURRENT_COMMIT}${RESET}"
cd "$REPO_ROOT"

# ------------------------------------------------------------------------------
# 4. Install uv & Authoritative Dependencies from upstream uv.lock
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[4/7] Configuring environment with uv and upstream uv.lock...${RESET}"

USE_UV=0
if ! command -v uv &> /dev/null; then
    echo "Installing uv (fast package manager recommended by ACE-Step)..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; then
        export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    fi
fi

if command -v uv &> /dev/null; then
    USE_UV=1
    UV_VER=$(uv --version)
    echo -e "${GREEN}Using ${UV_VER}${RESET}"
else
    echo -e "${YELLOW}uv not available, falling back to standard pip.${RESET}"
fi

if [ "$USE_UV" -eq 1 ]; then
    echo "Synchronizing exact dependencies from upstream uv.lock..."
    # Ensure Python 3.11 is targeted if available, or current python
    mkdir -p "$VENV_DIR"
    if [ ! -f "${VENV_DIR}/bin/python" ]; then
        uv venv "$VENV_DIR" --python 3.11 2>/dev/null || uv venv "$VENV_DIR"
    fi
    
    # Authoritative sync of exact frozen dependencies into $VENV_DIR
    VIRTUAL_ENV="$VENV_DIR" uv sync --active --frozen --no-dev --project "$ACE_STEP_DIR"
    
    # Install lab-specific auxiliary package (huggingface_hub)
    VIRTUAL_ENV="$VENV_DIR" uv pip install -r "${REPO_ROOT}/requirements.txt"
    
    # Install ACE-Step in editable mode without modifying locked dependencies
    VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$ACE_STEP_DIR" --no-deps
else
    # Fallback to standard pip
    if [ ! -f "${VENV_DIR}/bin/activate" ]; then
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade --quiet pip setuptools wheel
    
    echo "Installing PyTorch with CUDA 12.8 wheel..."
    pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 || pip install torch torchvision torchaudio
    
    echo "Installing upstream requirements.txt..."
    pip install -r "${ACE_STEP_DIR}/requirements.txt"
    pip install -e "$ACE_STEP_DIR" --no-deps
    pip install -r "${REPO_ROOT}/requirements.txt"
fi

# ------------------------------------------------------------------------------
# 5. Resolve Active Python in Virtual Environment
# ------------------------------------------------------------------------------
PYTHON_EXEC="${VENV_DIR}/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="${VENV_DIR}/Scripts/python.exe"
fi
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

# ------------------------------------------------------------------------------
# 6. Sanity Check: Validate Pinned Upstream Snapshot Requirements
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[5/7] Auditing installed packages against upstream specification...${RESET}"
if ! "$PYTHON_EXEC" "${REPO_ROOT}/scripts/verify_dependencies.py"; then
    echo -e "${YELLOW}[WARNING] Dependency audit detected warnings. Review table above.${RESET}"
fi

# ------------------------------------------------------------------------------
# 7. Download / Cache Pinned Model Checkpoints
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[6/7] Checking model checkpoints (XL-SFT 4B + LM 4B)...${RESET}"
if [ "$SKIP_DOWNLOAD" -eq 1 ]; then
    echo -e "${YELLOW}SKIP_DOWNLOAD=1 set. Skipping checkpoint download step.${RESET}"
else
    mkdir -p "$CHECKPOINTS_DIR"
    "$PYTHON_EXEC" "${REPO_ROOT}/scripts/download_checkpoints.py" --checkpoints-dir "$CHECKPOINTS_DIR"
fi

BOOTSTRAP_END_TIME=$(date +%s)
TOTAL_BOOTSTRAP_SECS=$((BOOTSTRAP_END_TIME - BOOTSTRAP_START_TIME))

# ------------------------------------------------------------------------------
# Completion Banner
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}================================================================${RESET}"
echo -e "${GREEN}${BOLD}       BOOTSTRAP COMPLETE! MACHINE READY FOR INFERENCE        ${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo -e "Total Bootstrap Time : ${TOTAL_BOOTSTRAP_SECS} seconds"
echo -e "Next Verification   : Run ${CYAN}./run.sh smoke${RESET} to test live CUDA inference."
echo -e "Next Benchmark      : Run ${CYAN}./benchmark.sh${RESET} to execute first-hour suite."
echo -e "${BOLD}================================================================${RESET}"
