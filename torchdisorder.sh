#!/bin/bash
#SBATCH --account=def-ctetsass
##SBATCH --partition=debug
#SBATCH --time=02-23:05
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
module load cuda
module load python/3.12
source /home/agore/scratch/torchdisorder/.venv/bin/activate
python scripts/cooper_test.py