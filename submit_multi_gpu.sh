#!/bin/bash
#SBATCH --job-name=td-multi
#SBATCH --account=def-cfeugmo
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4               # Request multiple GPUs
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@uwaterloo.ca

# ==============================================================================
# TorchDisorder Multi-GPU Training Script
# ==============================================================================
# 
# For large-scale structure optimization with multiple replicas
# Uses PyTorch DistributedDataParallel for multi-GPU training
#
# Usage:
#   sbatch submit_multi_gpu.sh
#
# ==============================================================================

echo "=============================================="
echo "TorchDisorder Multi-GPU Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p logs

# Load modules
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
module load nccl/2.18

# Activate environment
source $HOME/.venv/torchdisorder/bin/activate

# Environment setup
export PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Multi-GPU settings
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_GPUS_PER_NODE
export NCCL_DEBUG=INFO

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run with torchrun for distributed training (if supported)
# torchrun --nproc_per_node=$WORLD_SIZE scripts/train.py "$@"

# Or run standard single-GPU on first available
python scripts/train.py "$@"

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="
