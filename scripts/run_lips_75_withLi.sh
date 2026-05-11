#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — 75Li2S-25P2S5 Glass, Li-inclusive structural model
# =============================================================================
# Fits the full Li-P-S structure of 75Li2S-25P2S5 (= β-Li3PS4) glass to
# X-ray PDF data. Li atoms included with soft non-overlap penalty constraints.
# Supercell: 5×6×9 of Li3PS4_beta.
#
# Usage:
#   conda activate torchdisorder
#   ./scripts/run_lips_75_withLi.sh [--steps N]
#
# Outputs: outputs/LiPS_75_withLi_YYYY-MM-DD/HH-MM-SS/
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
echo "  TorchDisorder — 75Li2S-25P2S5  (with Li)"
echo "  Steps      : $STEPS"
echo "  Target     : S(Q)  (X-ray structure factor)"
echo "  Structure  : glass_75Li2S_withLi.cif"
echo "  Constraints: glass_75Li2S_withLi_constraints.json + Li penalty"
echo "============================================================"
echo ""

export PROJECT_ROOT="$(pwd)"
source "$PROJECT_ROOT/scripts/slurm_utils.sh"

log_hardware_info "75Li2S-25P2S5 Glass, Li-inclusive structural model" python logs

python scripts/train.py \
    experiment_name=LiPS_75_withLi \
    data=LiPS_75 \
    "data.json_path=\${data.root_dir}/json/glass_75Li2S_withLi_constraints.json" \
    structure=LiPS_75_withLi \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
