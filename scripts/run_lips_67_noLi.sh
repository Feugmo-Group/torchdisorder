#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — 67Li2S-33P2S5 Glass, Li-free structural model
# =============================================================================
# Fits the P-S sublattice of 67Li2S-33P2S5 glass to X-ray PDF data.
# Li atoms are excluded from the structural model; Faber-Ziman weights
# show that Li–X partial contributions are approximately self-cancelling
# for neutron S(Q), so the P-S sublattice captures the dominant signal.
#
# Usage:
#   conda activate torchdisorder
#   ./scripts/run_lips_67_noLi.sh [--steps N]
#
# Outputs: outputs/LiPS_67_noLi_YYYY-MM-DD/HH-MM-SS/
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
echo "  TorchDisorder — 67Li2S-33P2S5  (no Li)"
echo "  Steps      : $STEPS"
echo "  Target     : S(Q)  (X-ray structure factor)"
echo "  Structure  : glass_67Li2S_noLi.cif"
echo "  Constraints: glass_67Li2S_noLi_constraints.json"
echo "============================================================"
echo ""

export PROJECT_ROOT="$(pwd)"
source "$PROJECT_ROOT/scripts/slurm_utils.sh"

/opt/homebrew/Caskroom/miniconda/base/envs/torchdisorder/bin/python
 scripts/train.py \
    experiment_name=LiPS_67_noLi \
    data=LiPS_67 \
    structure=LiPS_67_noLi \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
