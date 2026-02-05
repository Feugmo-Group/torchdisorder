#!/bin/bash
# ==============================================================================
# TorchDisorder Interactive Session Script
# ==============================================================================
#
# Start an interactive GPU session for debugging and development.
#
# Usage:
#   ./interactive.sh           # 2 hour session
#   ./interactive.sh 4         # 4 hour session
#
# ==============================================================================

HOURS=${1:-2}

echo "Requesting interactive GPU session for ${HOURS} hours..."
echo ""

# For Compute Canada clusters (Graham, Cedar, Narval)
salloc --time=${HOURS}:00:00 \
       --account=def-cfeugmo \
       --gpus-per-node=1 \
       --cpus-per-task=8 \
       --mem=32G \
       --job-name=td-interactive

# Alternative: For clusters with partition-based GPU access
# salloc --time=${HOURS}:00:00 \
#        --partition=gpu \
#        --gres=gpu:1 \
#        --cpus-per-task=8 \
#        --mem=32G

# Once allocated, run:
# module load python/3.11 cuda/12.2 cudnn/8.9
# source $HOME/.venv/torchdisorder/bin/activate
# export PROJECT_ROOT=$(pwd)
# python scripts/train.py max_steps=1000  # Quick test
