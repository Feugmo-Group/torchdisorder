#!/bin/bash
# slurm_utils.sh – Shared helpers for hardware logging and run timing.
# Source this file after activating conda in every slurm script:
#   source "$PROJECT_ROOT/scripts/slurm_utils.sh"

# ---------------------------------------------------------------------------
# log_hardware_info  SYSTEM_LABEL  PYTHON_BIN  LOG_DIR
#   Prints and saves hardware details to logs/run_info_<JOBID>.txt
# ---------------------------------------------------------------------------
log_hardware_info() {
    local SYSTEM_LABEL="$1"
    local PYTHON_BIN="$2"
    local LOG_DIR="${3:-logs}"

    # GPU info via Python/torch
    local GPU_NAME GPU_VRAM GPU_DRIVER GPU_CUDA
    GPU_NAME=$($PYTHON_BIN -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name)
else:
    print('No CUDA GPU')
" 2>/dev/null || echo "N/A")

    GPU_VRAM=$($PYTHON_BIN -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'{p.total_memory // 1024**3} GB')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")

    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    GPU_CUDA=$($PYTHON_BIN -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")

    # CPU / RAM
    local CPU_MODEL RAM_TOTAL
    CPU_MODEL=$(lscpu 2>/dev/null | awk -F': +' '/Model name/{print $2; exit}' || echo "N/A")
    RAM_TOTAL=$(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo "N/A")

    # PyTorch version
    local TORCH_VER
    TORCH_VER=$($PYTHON_BIN -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")

    # Save to file
    mkdir -p "$LOG_DIR"
    INFO_FILE="$LOG_DIR/run_info_${SLURM_JOB_ID}.txt"
    cat > "$INFO_FILE" <<EOF
=== Hardware & Run Information ===
System      : $SYSTEM_LABEL
Job ID      : ${SLURM_JOB_ID:-local}
Node        : ${SLURM_NODELIST:-$(hostname)}
GPU device  : ${CUDA_VISIBLE_DEVICES:-SLURM-assigned}
GPU name    : $GPU_NAME
GPU VRAM    : $GPU_VRAM
GPU driver  : $GPU_DRIVER
CUDA        : $GPU_CUDA
CPU         : $CPU_MODEL
RAM         : $RAM_TOTAL
PyTorch     : $TORCH_VER
Start time  : $(date)
EOF

    echo ""
    echo "=== Hardware ==="
    cat "$INFO_FILE"
    echo "================"
    echo ""

    # Export for use by log_runtime
    export _RUN_START_TS=$(date +%s)
    export _RUN_INFO_FILE="$INFO_FILE"
}

# ---------------------------------------------------------------------------
# log_runtime
#   Appends end time and elapsed duration to the run_info file.
#   Call this at the very end of the training script.
# ---------------------------------------------------------------------------
log_runtime() {
    local END_TS=$(date +%s)
    local END_DATE=$(date)
    local ELAPSED=$(( END_TS - ${_RUN_START_TS:-$END_TS} ))
    local HH=$(( ELAPSED / 3600 ))
    local MM=$(( (ELAPSED % 3600) / 60 ))
    local SS=$(( ELAPSED % 60 ))

    cat >> "${_RUN_INFO_FILE:-/dev/null}" <<EOF
End time    : $END_DATE
Elapsed     : ${HH}h ${MM}m ${SS}s
Exit code   : $?
EOF

    echo ""
    echo "=== Timing ==="
    echo "  End     : $END_DATE"
    printf "  Elapsed : %dh %02dm %02ds\n" "$HH" "$MM" "$SS"
    echo "=============="
}
