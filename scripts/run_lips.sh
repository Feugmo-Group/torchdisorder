#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — Li-P-S Glass Training Run
# =============================================================================
#
# Fits the TorchDisorder model to 67Li2S-33P2S5 glass X-ray PDF data.
#
# Data  : data/xrd_measurements/Li3PS4/S_of_Q.csv  (X-ray, S(Q))
# Target: S_Q structure factor
# Steps : 5000 (adjust with --steps N below)
#
# Starting structure: glass_67Li2S_small.cif  (P-S sublattice, no Li)
# Constraints      : glass_67Li2S_small_constraints.json
#
# NOTE: This is the most computationally expensive run (~2–3× SiO2/GeO2)
#       because the LiPS data has more Q-bins (stride_q=4 → ~1930 bins).
#
# Usage:
#   conda activate torchdisorder
#   chmod +x scripts/run_lips.sh
#   ./scripts/run_lips.sh
#
# To change the number of steps:
#   ./scripts/run_lips.sh --steps 10000
#
# Outputs are written to:
#   outputs/LiPS_YYYY-MM-DD/HH-MM-SS/
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
echo "  TorchDisorder — Li-P-S Glass Run"
echo "  Steps      : $STEPS"
echo "  Target     : S(Q)  (X-ray structure factor)"
echo "  Data       : data/xrd_measurements/Li3PS4/S_of_Q.csv"
echo "  Structure  : data/crystal-structures/glass_67Li2S_small.cif"
echo "  Constraints: data/json/glass_67Li2S_small_constraints.json"
echo "  stride_q   : 4  (~1930 Q-bins from 7725 total)"
echo "============================================================"
echo ""

export PROJECT_ROOT="$(pwd)"

python scripts/train.py \
    experiment_name=LiPS \
    data=LiPS \
    structure=LiPS \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
