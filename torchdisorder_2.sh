#!/bin/bash
#SBATCH --account=def-ctetsass_gpu
#SBATCH --job-name=torchdisorder
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00

# --- FIR HARDWARE ALIGNMENT ---
# Request 1 full H100 using the specific syntax from docs
#SBATCH --gpus=h100:1

# Match the NUMA domain size (12 cores per GPU)
# Docs: "Launch 4 tasks per node... --cpus-per-task=12"
# Since we launch 1 task, we take 1 NUMA block (12 cores)
#SBATCH --cpus-per-task=12

# Memory: 1/4 of the node (~280G available). Requesting 96G is safe.
#SBATCH --mem=280G
# ------------------------------
#SBATCH --mail-type=END,FAIL,BEGIN                        # Email on completion/failure
#SBATCH --mail-user=agore@uwaterloo.ca                 # Email address
source /scratch/agore/torchdisorder/ENV/bin/activate

export PYTHONPATH=$PYTHONPATH:/scratch/agore/torchdisorder

python scripts/cooper_test_2.py




