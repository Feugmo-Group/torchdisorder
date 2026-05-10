#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

STEPS=5000
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "============================================================"
echo "  TorchDisorder — 75Li2S-25P2S5  small  (no Li)"
echo "  Steps : $STEPS"
echo "============================================================"

export PROJECT_ROOT="$(pwd)"

python scripts/train.py \
    experiment_name=LiPS_75_noLi_small \
    data=LiPS_75 \
    structure=LiPS_75_noLi_small \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100 \
    "${EXTRA_ARGS[@]}"
