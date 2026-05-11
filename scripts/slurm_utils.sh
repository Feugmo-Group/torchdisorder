#!/bin/bash
# slurm_utils.sh – Shared helpers for hardware logging and run timing.
# Works for both SLURM jobs and direct bash runs.
#
# Usage in any run script (after PROJECT_ROOT is set):
#   source "$PROJECT_ROOT/scripts/slurm_utils.sh"
#   log_hardware_info "System label" "python_binary" "log_dir"
#   # ... training command ...
#   # log_runtime is called automatically via EXIT trap

# ---------------------------------------------------------------------------
# _run_label: unique label for the log file
#   SLURM jobs  → job ID
#   direct runs → timestamp
# ---------------------------------------------------------------------------
_run_label() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "$SLURM_JOB_ID"
    else
        date +%Y%m%d_%H%M%S
    fi
}

# ---------------------------------------------------------------------------
# log_hardware_info  SYSTEM_LABEL  PYTHON_BIN  LOG_DIR
#   Prints and saves hardware details. Also registers log_runtime via EXIT
#   trap so timing is captured even if training fails.
# ---------------------------------------------------------------------------
log_hardware_info() {
    local SYSTEM_LABEL="$1"
    local PYTHON_BIN="${2:-python}"
    local LOG_DIR="${3:-logs}"

    # GPU info
    local GPU_NAME GPU_VRAM GPU_DRIVER GPU_CUDA
    GPU_NAME=$($PYTHON_BIN -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(0).name)
else:
    print('CPU only')
" 2>/dev/null || echo "N/A")

    GPU_VRAM=$($PYTHON_BIN -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f'{gb} GB')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")

    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    GPU_CUDA=$($PYTHON_BIN -c "import torch; print(torch.version.cuda or 'N/A')" 2>/dev/null || echo "N/A")

    # CPU / RAM
    local CPU_MODEL RAM_TOTAL
    CPU_MODEL=$(lscpu 2>/dev/null | awk -F': +' '/Model name/{print $2; exit}' || echo "N/A")
    RAM_TOTAL=$(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo "N/A")

    # PyTorch version
    local TORCH_VER
    TORCH_VER=$($PYTHON_BIN -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")

    # Build log file path
    mkdir -p "$LOG_DIR"
    export _RUN_INFO_FILE="$LOG_DIR/run_info_$(_run_label).txt"
    export _RUN_START_TS=$(date +%s)
    export _RUN_PYTHON_BIN="$PYTHON_BIN"

    cat > "$_RUN_INFO_FILE" <<EOF
=== Hardware & Run Information ===
System      : $SYSTEM_LABEL
Job ID      : ${SLURM_JOB_ID:-"(direct run)"}
Node        : ${SLURM_NODELIST:-$(hostname)}
GPU device  : ${CUDA_VISIBLE_DEVICES:-"(SLURM-assigned or default)"}
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
    cat "$_RUN_INFO_FILE"
    echo "================"
    echo ""

    # Register timing automatically — fires on exit, pass/fail
    trap log_runtime EXIT
}

# ---------------------------------------------------------------------------
# log_runtime
#   Appends end time, elapsed duration, and exit code to the run_info file.
#   Called automatically via EXIT trap set by log_hardware_info.
# ---------------------------------------------------------------------------
log_runtime() {
    # Unregister trap to avoid double-call
    trap - EXIT

    local EXIT_CODE=$?
    local END_TS=$(date +%s)
    local END_DATE=$(date)
    local ELAPSED=$(( END_TS - ${_RUN_START_TS:-$END_TS} ))
    local HH=$(( ELAPSED / 3600 ))
    local MM=$(( (ELAPSED % 3600) / 60 ))
    local SS=$(( ELAPSED % 60 ))

    if [[ -n "${_RUN_INFO_FILE:-}" ]]; then
        cat >> "$_RUN_INFO_FILE" <<EOF
End time    : $END_DATE
Elapsed     : ${HH}h ${MM}m ${SS}s
Exit code   : $EXIT_CODE
EOF
    fi

    echo ""
    echo "=== Timing ==="
    echo "  End     : $END_DATE"
    printf "  Elapsed : %dh %02dm %02ds\n" "$HH" "$MM" "$SS"
    [[ $EXIT_CODE -ne 0 ]] && echo "  Exit    : $EXIT_CODE (FAILED)"
    echo "=============="

    return $EXIT_CODE
}
