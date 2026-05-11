#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — SiO2 Training Run with MACE Regularization
# =============================================================================
#
# Fits TorchDisorder to vitreous silica neutron diffraction data with an
# optional MACE force-penalty and periodic FIRE relaxation to keep structures
# physically reasonable during optimisation.
#
# Data   : data/xrd_measurements/SiO2/F_of_Q.csv  (Kohara 2005, neutron)
# Target : F(Q) reduced structure factor
# Steps  : 500 by default (short smoke-test; increase for production)
#
# Usage:
#   conda activate torchdisorder
#   chmod +x scripts/run_sio2_mace.sh
#   bash scripts/run_sio2_mace.sh
#
# Options:
#   --steps    N    number of optimisation steps          (default: 500)
#   --model    STR  MACE model size: small|medium|large   (default: small)
#   --fw       F    force penalty weight λ                (default: 1e-3)
#   --every    N    apply force penalty every N steps     (default: 10)
#   --relax    N    FIRE relaxation every N steps (0=off) (default: 100)
#   --cpu           force CPU even when GPU is available
#   Extra positional/keyword arguments are forwarded to train.py as-is.
#
# Examples:
#   # Quick smoke-test (CPU, MACE-small, 500 steps)
#   bash scripts/run_sio2_mace.sh
#
#   # Production GPU run (2000 steps, stronger force weight)
#   bash scripts/run_sio2_mace.sh --steps 2000 --fw 5e-3
#
#   # Disable MACE (falls back to plain SiO2 run)
#   bash scripts/run_sio2_mace.sh mlip.enabled=false
#
# Outputs are written to:
#   outputs/SiO2_mace_YYYY-MM-DD/HH-MM-SS/
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Parse arguments ───────────────────────────────────────────────────────────
STEPS=500
MACE_MODEL="small"
FORCE_WEIGHT="1e-3"
APPLY_EVERY=10
RELAX_EVERY=100
ACCELERATOR=""          # empty = auto-detect below
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)  STEPS="$2";        shift 2 ;;
        --model)  MACE_MODEL="$2";   shift 2 ;;
        --fw)     FORCE_WEIGHT="$2"; shift 2 ;;
        --every)  APPLY_EVERY="$2";  shift 2 ;;
        --relax)  RELAX_EVERY="$2";  shift 2 ;;
        --cpu)    ACCELERATOR="cpu"; shift   ;;
        *)        EXTRA_ARGS+=("$1"); shift  ;;
    esac
done

# ── Auto-detect best accelerator ─────────────────────────────────────────────
# MPS is skipped: torch_sim's JIT-compiled neighbor code is incompatible with
# the MPS linalg.inv kernel on non-contiguous tensors (PyTorch bug).
if [[ -z "$ACCELERATOR" ]]; then
    ACCELERATOR=$(python -c "
import torch
if torch.cuda.is_available():
    print('cuda')
else:
    print('cpu')
" 2>/dev/null || echo "cpu")
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  TorchDisorder — SiO2 + MACE Regularization"
echo "  Steps        : $STEPS"
echo "  Accelerator  : $ACCELERATOR"
echo "  Target       : F(Q)  (neutron reduced structure factor)"
echo "  Data         : data/xrd_measurements/SiO2/F_of_Q.csv"
echo "  Structure    : data/crystal-structures/sio2_glass.cif"
echo "  MACE model   : $MACE_MODEL"
echo "  Force weight : $FORCE_WEIGHT  (every $APPLY_EVERY steps)"
echo "  FIRE relax   : every $RELAX_EVERY steps"
echo "============================================================"
echo ""

# ── Hardware logging ──────────────────────────────────────────────────────────
export PROJECT_ROOT="$(pwd)"
source "$PROJECT_ROOT/scripts/slurm_utils.sh"

log_hardware_info "SiO2 + MACE Regularization" python logs

# ── Training ──────────────────────────────────────────────────────────────────
python scripts/train.py \
    experiment_name=SiO2_mace \
    data=SiO2 \
    structure=silica \
    target=F_Q \
    max_steps="$STEPS" \
    accelerator="$ACCELERATOR" \
    output.plot_interval=50 \
    mlip.enabled=true \
    mlip.backend=mace \
    mlip.model="$MACE_MODEL" \
    mlip.force_weight="$FORCE_WEIGHT" \
    mlip.apply_every="$APPLY_EVERY" \
    mlip.relax_every="$RELAX_EVERY" \
    mlip.relax_max_steps=20 \
    "${EXTRA_ARGS[@]}"
