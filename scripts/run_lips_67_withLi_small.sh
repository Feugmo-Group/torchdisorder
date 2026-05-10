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
echo "  TorchDisorder — 67Li2S-33P2S5  small  (with Li)"
echo "  Steps : $STEPS"
echo "============================================================"

export PROJECT_ROOT="$(pwd)"

python scripts/train.py \
    experiment_name=LiPS_67_withLi_small \
    data=LiPS_67 \
    "data.json_path=${PROJECT_ROOT}/data/json/glass_67Li2S_small_withLi_constraints.json" \
    structure=LiPS_67_withLi_small \
    target=S_Q \
    max_steps="$STEPS" \
    accelerator=cpu \
    output.plot_interval=100 \
    "${EXTRA_ARGS[@]}"
