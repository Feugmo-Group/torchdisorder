#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — SiO2 Training Run
# =============================================================================
#
# Fits the TorchDisorder model to vitreous silica neutron diffraction data.
#
# Data  : data/xrd_measurements/SiO2/F_of_Q.csv  (Kohara 2005, neutron)
# Target: F(Q) reduced structure factor
# Steps : 5000 (adjust with --steps N below)
#
# Usage:
#   conda activate torchdisorder
#   chmod +x scripts/run_sio2.sh
#   ./scripts/run_sio2.sh
#
# To change the number of steps:
#   ./scripts/run_sio2.sh --steps 10000
#
# Outputs are written to:
#   outputs/SiO2_YYYY-MM-DD/HH-MM-SS/
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Parse optional --steps argument ──────────────────────────────────────────
STEPS=5000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "  TorchDisorder — SiO2 Run"
echo "  Steps      : $STEPS"
echo "  Target     : F(Q)  (neutron reduced structure factor)"
echo "  Data       : data/xrd_measurements/SiO2/F_of_Q.csv"
echo "  Structure  : data/crystal-structures/sio2_glass.cif"
echo "  Constraints: data/json/sio2_glass_constraints.json"
echo "============================================================"
echo ""

export PROJECT_ROOT="$(pwd)"
source "$PROJECT_ROOT/scripts/slurm_utils.sh"

log_hardware_info "SiO2 Training" python logs

python scripts/train.py \
    experiment_name=SiO2 \
    data=SiO2 \
    structure=silica \
    target=F_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
