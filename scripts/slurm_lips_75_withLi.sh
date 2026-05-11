#!/bin/bash
#SBATCH --job-name=lips75Li
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lips_75_withLi_%j.out
#SBATCH --error=logs/lips_75_withLi_%j.err

set -e

# --- 1. Environment Protection ---
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
PROJECT_ROOT="/home/conrard/torchdisorder"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# --- 2. Directory Setup ---
cd "$PROJECT_ROOT"
mkdir -p logs

# --- 3. Logging Info ---
echo "=================================================="
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $SLURM_NODELIST"
echo "  GPU       : $CUDA_VISIBLE_DEVICES"
echo "  Start     : $(date)"
echo "  System    : 75Li2S-25P2S5, with Li"
echo "  Work Dir  : $(pwd)"
echo "=================================================="

# --- 4. Load Software Stack ---
module load nvhpc26.3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torchdisorder

CONDA_PYTHON="/home/conrard/.conda/envs/torchdisorder/bin/python"
$CONDA_PYTHON -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# --- 5. Training Execution ---
export WANDB_MODE=offline

echo "Starting Training..."

$CONDA_PYTHON scripts/train.py \
    experiment_name=LiPS_75_withLi \
    data=LiPS_75 \
    "data.json_path=$PROJECT_ROOT/data/json/glass_75Li2S_withLi_constraints.json" \
    structure=LiPS_75_withLi \
    target=S_Q \
    max_steps=5000 \
    accelerator=cuda \
    output.plot_interval=500

echo "=================================================="
echo "  Done: $(date)"
echo "=================================================="
