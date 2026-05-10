#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — 75Li2S-25P2S5 Glass, Li-free structural model
# =============================================================================
# Fits the P-S sublattice of 75Li2S-25P2S5 (= β-Li3PS4) glass to X-ray PDF.
# Supercell: 5×6×9 of Li3PS4_beta.
#
# Usage:
#   conda activate torchdisorder
#   ./scripts/run_lips_75_noLi.sh [--steps N]
#
# Outputs: outputs/LiPS_75_noLi_YYYY-MM-DD/HH-MM-SS/
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
echo "  TorchDisorder — 75Li2S-25P2S5  (no Li)"
echo "  Steps      : $STEPS"
echo "  Target     : S(Q)  (X-ray structure factor)"
echo "  Structure  : glass_75Li2S_noLi.cif"
echo "  Constraints: glass_75Li2S_noLi_constraints.json"
echo "============================================================"
echo ""

export PROJECT_ROOT="$(pwd)"

python scripts/train.py \
    experiment_name=LiPS_75_noLi \
    data=LiPS_75 \
    structure=LiPS_75_noLi \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
