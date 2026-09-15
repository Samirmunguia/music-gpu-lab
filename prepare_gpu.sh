#!/usr/bin/env bash
# ==============================================================================
# prepare_gpu.sh - One-Command GPU Readiness & Verification Pipeline
# Chains: verify_gpu.sh -> bootstrap.sh -> run.sh smoke
# Stops immediately on any failure. Leaves instance 100% benchmark-ready.
# ==============================================================================

set -eo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
RESET="\033[0m"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}         music-gpu-lab :: One-Command GPU Preparation           ${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo -e "This script will sequentially verify hardware, configure dependencies,"
echo -e "download pinned checkpoints, and execute a live CUDA smoke test.\n"

# ------------------------------------------------------------------------------
# STEP 1: Hardware & Environment Verification
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}>>> STEP 1/3: AUDITING GPU HARDWARE & ENVIRONMENT...${RESET}"
if ! ./verify_gpu.sh; then
    echo -e "\n${RED}[ABORT] GPU verification failed! Fix the errors above before proceeding.${RESET}"
    exit 1
fi
echo -e "${GREEN}[OK] Step 1 passed: NVIDIA GPU, CUDA, RAM, and storage validated.${RESET}\n"

# ------------------------------------------------------------------------------
# STEP 2: Bootstrap Setup & Weight Check
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}>>> STEP 2/3: BOOTSTRAPPING DEPENDENCIES & MODEL CHECKPOINTS...${RESET}"
if ! ./bootstrap.sh; then
    echo -e "\n${RED}[ABORT] Bootstrap setup failed! Check error messages above.${RESET}"
    exit 1
fi
echo -e "${GREEN}[OK] Step 2 passed: Virtualenv, pinned ACE-Step, and weights verified.${RESET}\n"

# ------------------------------------------------------------------------------
# STEP 3: Live Pipeline Smoke Test
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}>>> STEP 3/3: EXECUTING LIVE GPU PIPELINE SMOKE TEST...${RESET}"
if ! ./run.sh smoke; then
    echo -e "\n${RED}[ABORT] Smoke test failed! Live CUDA inference was not successful.${RESET}"
    exit 1
fi
echo -e "${GREEN}[OK] Step 3 passed: Real CUDA inference and audio output validated.${RESET}\n"

# ------------------------------------------------------------------------------
# Final Ready Banner
# ------------------------------------------------------------------------------
echo -e "${BOLD}================================================================${RESET}"
echo -e "${GREEN}${BOLD}     SUCCESS: INSTANCE IS FULLY VERIFIED & READY TO BENCHMARK   ${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo -e "You can now launch the automated first-hour benchmark suite:"
echo -e "  ${CYAN}./benchmark.sh${RESET}"
echo -e ""
echo -e "Or run individual style evaluations:"
echo -e "  ${CYAN}./run.sh ace-step reggaeton_nocturno instrumental --seeds 4${RESET}"
echo -e "  ${CYAN}./run.sh ace-step dembow_dominicano male-vocal --seeds 2${RESET}"
echo -e "${BOLD}================================================================${RESET}"
