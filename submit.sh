#!/bin/bash
#SBATCH --job-name=torchdisorder
#SBATCH --account=def-cfeugmo          # Replace with your allocation
#SBATCH --time=24:00:00                 # Max walltime (HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1               # Request 1 GPU
#SBATCH --mem=32G                       # Memory per node
#SBATCH --output=logs/%x_%j.out         # stdout file
#SBATCH --error=logs/%x_%j.err          # stderr file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@uwaterloo.ca  # Replace with your email

# ==============================================================================
# TorchDisorder SLURM Submission Script
# ==============================================================================
# 
# Usage:
#   sbatch submit.sh                          # Run with defaults
#   sbatch submit.sh data=GeO2                # Override config
#   sbatch submit.sh max_steps=100000         # Custom max steps
#
# For Compute Canada clusters:
#   sbatch --partition=gpu submit.sh          # Graham, Cedar, Narval
#
# ==============================================================================

echo "=============================================="
echo "TorchDisorder Training Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust for your cluster)
# ============================================
# Compute Canada (Graham, Cedar, Narval, Beluga)
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Alternative for other clusters:
# module load anaconda3
# module load cuda/12.0

# Activate virtual environment
# ============================================
# Option 1: Using virtualenv
source $HOME/.venv/torchdisorder/bin/activate

# Option 2: Using conda
# conda activate torchdisorder

# Option 3: Using Poetry (recommended)
# export POETRY_VIRTUALENVS_PATH=$HOME/.cache/pypoetry/virtualenvs
# source $(poetry env info --path)/bin/activate

# Set environment variables
# ============================================
export PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Disable W&B if no internet (common on compute nodes)
# export WANDB_MODE=offline

# For debugging
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

# Print environment info
# ============================================
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Run the training script
# ============================================
# Pass all script arguments to train.py
python scripts/train.py "$@"

# Capture exit code
EXIT_CODE=$?

# Print completion info
# ============================================
echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
