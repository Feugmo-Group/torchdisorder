#!/usr/bin/env bash
# =============================================================================
# TorchDisorder — GeO2 Training Run
# =============================================================================
#
# Fits the TorchDisorder model to vitreous germania neutron diffraction data.
#
# Data  : data/xrd_measurements/GeO2/F_of_Q.csv  (Kohara 2005, neutron)
# Target: F(Q) reduced structure factor
# Steps : 5000 (adjust with --steps N below)
#
# Usage:
#   conda activate torchdisorder
#   chmod +x scripts/run_geo2.sh
#   ./scripts/run_geo2.sh
#
# To change the number of steps:
#   ./scripts/run_geo2.sh --steps 10000
#
# Outputs are written to:
#   outputs/GeO2_YYYY-MM-DD/HH-MM-SS/
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
echo "  TorchDisorder — GeO2 Run"
echo "  Steps      : $STEPS"
echo "  Target     : F(Q)  (neutron reduced structure factor)"
echo "  Data       : data/xrd_measurements/GeO2/F_of_Q.csv"
echo "  Structure  : data/crystal-structures/geo2_glass.cif"
echo "  Constraints: data/json/geo2_glass_constraints.json"
echo "============================================================"
echo ""

export PROJECT_ROOT="$(pwd)"

python scripts/train.py \
    experiment_name=GeO2 \
    data=GeO2 \
    structure=GeO2 \
    target=F_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
