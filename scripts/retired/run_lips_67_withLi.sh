#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — 67Li2S-33P2S5 Glass, Li-inclusive structural model
# =============================================================================
# Fits the full Li-P-S structure of 67Li2S-33P2S5 glass to X-ray PDF data.
# Li atoms are included with soft non-overlap penalty constraints (Li-S ≥ 2.3 Å,
# Li-P ≥ 2.8 Å, Li-Li ≥ 2.0 Å) from the li_constraints block in the JSON.
# Li is excluded from P order-parameter neighbour search (element_filter=[P,S]).
#
# Usage:
#   conda activate torchdisorder
#   ./scripts/run_lips_67_withLi.sh [--steps N]
#
# Outputs: outputs/LiPS_67_withLi_YYYY-MM-DD/HH-MM-SS/
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

STEPS=5000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "  TorchDisorder — 67Li2S-33P2S5  (with Li)"
echo "  Steps      : $STEPS"
echo "  Target     : S(Q)  (X-ray structure factor)"
echo "  Structure  : glass_67Li2S_withLi.cif"
echo "  Constraints: glass_67Li2S_withLi_constraints.json + Li penalty"
echo "============================================================"
echo ""

export PROJECT_ROOT="$(pwd)"
source "$PROJECT_ROOT/scripts/slurm_utils.sh"

log_hardware_info "67Li2S-33P2S5 Glass, Li-inclusive structural model" python logs

python scripts/train.py \
    experiment_name=LiPS_67_withLi \
    data=LiPS_67 \
    "data.json_path=\${data.root_dir}/json/glass_67Li2S_withLi_constraints.json" \
    structure=LiPS_67_withLi \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
