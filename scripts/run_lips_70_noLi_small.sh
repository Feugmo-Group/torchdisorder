#!/usr/bin/env bash
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

echo "============================================================"
echo "  TorchDisorder — 70Li2S-30P2S5  small  (no Li)"
echo "  Steps : $STEPS"
echo "============================================================"

export PROJECT_ROOT="$(pwd)"

python scripts/train.py \
    experiment_name=LiPS_70_noLi_small \
    data=LiPS_70 \
    structure=LiPS_70_noLi_small \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100
