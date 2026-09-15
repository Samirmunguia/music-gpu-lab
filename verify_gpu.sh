#!/usr/bin/env bash
# ==============================================================================
# verify_gpu.sh - Pre-flight GPU & Environment Validation for music-gpu-lab
# Validates: NVIDIA GPU, VRAM, CUDA, PyTorch CUDA visibility, RAM, Disk, Network
# ==============================================================================

set -u

# Terminal styling
BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
RESET="\033[0m"

PASS="${GREEN}[ PASS ]${RESET}"
FAIL="${RED}[ FAIL ]${RESET}"
WARN="${YELLOW}[ WARN ]${RESET}"
INFO="${CYAN}[ INFO ]${RESET}"

TOTAL_ERRORS=0

echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}         music-gpu-lab :: Pre-Flight Environment Audit         ${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo -e "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")\n"

# ------------------------------------------------------------------------------
# 1. NVIDIA Driver & GPU Hardware
# ------------------------------------------------------------------------------
echo -e "${BOLD}1. NVIDIA Hardware & Driver Detection${RESET}"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo "unknown")
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    
    echo -e "   Driver Version : ${DRIVER_VER}"
    echo -e "   GPUs Detected  : ${GPU_COUNT}"
    while IFS= read -r gpu; do
        echo -e "   GPU Model      : ${BOLD}${gpu}${RESET}"
    done <<< "$GPU_NAMES"
    echo -e "   Status         : ${PASS} NVIDIA Driver and Hardware accessible"
else
    echo -e "   Status         : ${FAIL} 'nvidia-smi' not found! No NVIDIA driver or GPU detected."
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
fi

# ------------------------------------------------------------------------------
# 2. VRAM Capacity & Free Memory
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}2. GPU Memory (VRAM)${RESET}"
if command -v nvidia-smi &> /dev/null; then
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)
    FREE_VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)
    
    TOTAL_VRAM_GB=$(awk "BEGIN {printf \"%.1f\", ${TOTAL_VRAM_MB} / 1024}")
    FREE_VRAM_GB=$(awk "BEGIN {printf \"%.1f\", ${FREE_VRAM_MB} / 1024}")
    
    echo -e "   Total VRAM     : ${TOTAL_VRAM_GB} GB (${TOTAL_VRAM_MB} MiB)"
    echo -e "   Free VRAM      : ${FREE_VRAM_GB} GB (${FREE_VRAM_MB} MiB)"
    
    if [ "${TOTAL_VRAM_MB}" -ge 90000 ]; then
        echo -e "   Suitability    : ${PASS} Massive VRAM (>= 90GB) - RTX PRO 6000 Blackwell (96 GB) / H200 class"
    elif [ "${TOTAL_VRAM_MB}" -ge 70000 ]; then
        echo -e "   Suitability    : ${PASS} Optimal for unquantized XL-SFT (4B DiT + 4B LM) - A100 80GB class"
    elif [ "${TOTAL_VRAM_MB}" -ge 40000 ]; then
        echo -e "   Suitability    : ${PASS} Excellent for XL-SFT - RTX 6000 Ada (48 GB) / A100 40GB class"
    elif [ "${TOTAL_VRAM_MB}" -ge 20000 ]; then
        echo -e "   Suitability    : ${WARN} Viable (24GB VRAM). Recommend CPU offload if OOM occurs."
    else
        echo -e "   Suitability    : ${WARN} Low VRAM (< 20GB). XL-SFT 4B models may require CPU offload or quantization."
    fi
else
    echo -e "   Status         : ${FAIL} Cannot measure VRAM (nvidia-smi unavailable)"
fi

# ------------------------------------------------------------------------------
# 3. CUDA Availability & Toolchain
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}3. CUDA Runtime & Toolchain${RESET}"
CUDA_VER_SMI="N/A"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER_SMI=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}' || echo "N/A")
    echo -e "   Driver CUDA Cap: ${CUDA_VER_SMI}"
fi

if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk -F'release ' '{print $2}' | awk -F',' '{print $1}')
    echo -e "   NVCC Compiler  : ${NVCC_VER} (${PASS} nvcc present)"
else
    echo -e "   NVCC Compiler  : ${INFO} nvcc not found in PATH (standard runtime image, PyTorch wheels bundled)"
fi

# ------------------------------------------------------------------------------
# 4. PyTorch CUDA Visibility
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}4. PyTorch CUDA Visibility${RESET}"
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
fi

if command -v "$PYTHON_CMD" &> /dev/null; then
    PYTORCH_STATUS=$("$PYTHON_CMD" -c "
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    dev_count = torch.cuda.device_count()
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda
    name = torch.cuda.get_device_name(0) if cuda_avail else 'None'
    print(f'OK|{torch_ver}|{cuda_avail}|{dev_count}|{cuda_ver}|{name}')
except ImportError:
    print('NO_TORCH')
except Exception as e:
    print(f'ERR|{e}')
" 2>/dev/null || echo "FAILED")

    if [ "$PYTORCH_STATUS" = "NO_TORCH" ]; then
        echo -e "   PyTorch Status : ${WARN} PyTorch not installed yet (will be installed by ./bootstrap.sh)"
    elif [[ "$PYTORCH_STATUS" == OK* ]]; then
        IFS='|' read -r _ TORCH_V CUDA_AV DEV_C CUDA_V DEV_N <<< "$PYTORCH_STATUS"
        echo -e "   PyTorch Version: ${TORCH_V}"
        echo -e "   PyTorch CUDA   : ${CUDA_V}"
        echo -e "   Visible Devices: ${DEV_C}"
        echo -e "   Primary Device : ${DEV_N}"
        if [ "$CUDA_AV" = "True" ]; then
            echo -e "   CUDA Available : ${PASS} PyTorch can execute kernels on GPU"
        else
            echo -e "   CUDA Available : ${FAIL} torch.cuda.is_available() is False"
            TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
        fi
    else
        echo -e "   PyTorch Status : ${WARN} Could not inspect PyTorch: ${PYTORCH_STATUS}"
    fi
else
    echo -e "   Python Status  : ${WARN} Python3 not found in PATH"
fi

# ------------------------------------------------------------------------------
# 5. System Memory (RAM)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}5. Host System RAM${RESET}"
if [ -f /proc/meminfo ]; then
    TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    AVAIL_RAM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    TOTAL_RAM_GB=$(awk "BEGIN {printf \"%.1f\", ${TOTAL_RAM_KB} / 1024 / 1024}")
    AVAIL_RAM_GB=$(awk "BEGIN {printf \"%.1f\", ${AVAIL_RAM_KB} / 1024 / 1024}")
    echo -e "   Total RAM      : ${TOTAL_RAM_GB} GB"
    echo -e "   Available RAM  : ${AVAIL_RAM_GB} GB"
    if [ "$TOTAL_RAM_KB" -ge 32000000 ]; then
        echo -e "   Status         : ${PASS} Host RAM sufficient (>= 32GB recommended for 4B models)"
    elif [ "$TOTAL_RAM_KB" -ge 16000000 ]; then
        echo -e "   Status         : ${WARN} Host RAM moderate (16-32GB)"
    else
        echo -e "   Status         : ${WARN} Host RAM low (< 16GB). Model loading may be tight."
    fi
elif command -v free &> /dev/null; then
    free -h
    echo -e "   Status         : ${INFO} Memory inspected via 'free'"
else
    echo -e "   Status         : ${INFO} Unable to inspect /proc/meminfo"
fi

# ------------------------------------------------------------------------------
# 6. Disk Storage Capacity
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}6. Storage & Disk Space${RESET}"
CHECKPOINTS_TARGET="${CHECKPOINTS_DIR:-$(pwd)}"
if command -v df &> /dev/null; then
    DISK_AVAIL_KB=$(df -k "$CHECKPOINTS_TARGET" | tail -n 1 | awk '{print $4}')
    DISK_AVAIL_GB=$(awk "BEGIN {printf \"%.1f\", ${DISK_AVAIL_KB} / 1024 / 1024}")
    DISK_TARGET_PATH=$(df -P "$CHECKPOINTS_TARGET" | tail -n 1 | awk '{print $6}')
    echo -e "   Filesystem Path: ${CHECKPOINTS_TARGET} (mounted on ${DISK_TARGET_PATH})"
    echo -e "   Available Space: ${DISK_AVAIL_GB} GB"
    if [ "$DISK_AVAIL_KB" -ge 50000000 ]; then
        echo -e "   Status         : ${PASS} Disk space ample (>= 50GB available)"
    elif [ "$DISK_AVAIL_KB" -ge 30000000 ]; then
        echo -e "   Status         : ${WARN} Disk space adequate (~30-50GB). Keep an eye on checkpoint storage."
    else
        echo -e "   Status         : ${FAIL} Insufficient disk space (< 30GB). 4B DiT + 4B LM checkpoints need ~35GB."
        TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
    fi
else
    echo -e "   Status         : ${INFO} 'df' command not available"
fi

# ------------------------------------------------------------------------------
# 7. Basic Network & Download Readiness
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}7. Network Connectivity (Model Registries)${RESET}"
if command -v curl &> /dev/null; then
    if curl -sSfI --connect-timeout 4 https://huggingface.co > /dev/null 2>&1; then
        echo -e "   HuggingFace    : ${PASS} Reachable (https://huggingface.co)"
    else
        echo -e "   HuggingFace    : ${FAIL} Unreachable! Check outbound internet or proxy."
        TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
    fi

    if curl -sSfI --connect-timeout 4 https://github.com > /dev/null 2>&1; then
        echo -e "   GitHub         : ${PASS} Reachable (https://github.com)"
    else
        echo -e "   GitHub         : ${WARN} Unreachable (https://github.com)"
    fi
else
    echo -e "   Status         : ${WARN} 'curl' not found, unable to verify internet connectivity"
fi

# ------------------------------------------------------------------------------
# Final Summary & Exit Verdict
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}================================================================${RESET}"
if [ "$TOTAL_ERRORS" -eq 0 ]; then
    echo -e "${BOLD}AUDIT VERDICT: ${GREEN}READY FOR SETUP & INFERENCE${RESET}"
    echo -e "Next step: Run ${CYAN}./bootstrap.sh${RESET} to configure environment and weights."
    echo -e "${BOLD}================================================================${RESET}"
    exit 0
else
    echo -e "${BOLD}AUDIT VERDICT: ${RED}FAILED (${TOTAL_ERRORS} blocking errors detected)${RESET}"
    echo -e "Check the log items marked ${FAIL} above before proceeding."
    echo -e "${BOLD}================================================================${RESET}"
    exit 1
fi
