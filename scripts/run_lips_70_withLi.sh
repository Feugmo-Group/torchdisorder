#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — 70Li2S-30P2S5 Glass, Li-inclusive structural model
# =============================================================================
# Fits the full Li-P-S structure of 70Li2S-30P2S5 glass to X-ray PDF data.
# Li atoms included with soft non-overlap penalty constraints.
#
# Usage:
#   conda activate torchdisorder
#   ./scripts/run_lips_70_withLi.sh [--steps N]
#
# Outputs: outputs/LiPS_70_withLi_YYYY-MM-DD/HH-MM-SS/
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
echo "  TorchDisorder — 70Li2S-30P2S5  (with Li)"
echo "  Steps      : $STEPS"
echo "  Target     : S(Q)  (X-ray structure factor)"
echo "  Structure  : glass_70Li2S_withLi.cif"
echo "  Constraints: glass_70Li2S_withLi_constraints.json + Li penalty"
echo "============================================================"
echo ""

export PROJECT_ROOT="$(pwd)"

python scripts/train.py \
    experiment_name=LiPS_70_withLi \
    data=LiPS_70 \
    "data.json_path=\${data.root_dir}/json/glass_70Li2S_withLi_constraints.json" \
    structure=LiPS_70_withLi \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
