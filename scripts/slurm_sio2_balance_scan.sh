#!/bin/bash
# =============================================================================
# TorchDisorder — SiO2 constraint/chi2 balancing scan
# =============================================================================
#
# Answers: which balancing strategy, and which setting, produces the most
# physical structure -- judged on quantities the refinement does NOT fit.
#
# Every variant refines the same published GAP glass against the same F(Q).
# The only thing that changes is how the constraint penalty rho is chosen.
# Scoring is done afterwards by scripts/compare_to_literature.py on bond-length
# spread, O-Si-O and Si-O-Si angles and second-neighbour shells -- none of which
# enter the objective, so they are an independent test rather than a restatement
# of it.
#
# Submit:
#     sbatch scripts/slurm_sio2_balance_scan.sh
# Watch:
#     squeue -u $USER;  tail -f logs/sio2_scan_<jobid>_<task>.out
# Score once finished:
#     bash scripts/score_balance_scan.sh
#
# 6 variants, 4 GPUs -> %4 concurrency.
# =============================================================================
#SBATCH --job-name=sio2scan
#SBATCH --array=0-5%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sio2_scan_%A_%a.out
#SBATCH --error=logs/sio2_scan_%A_%a.err

set -e

# --- 1. Environment protection ---
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
PROJECT_ROOT="/home/conrard/torchdisorder"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD

cd "$PROJECT_ROOT"
mkdir -p logs

# --- 2. Variant table -------------------------------------------------------
# Each entry: <label> <hydra overrides...>
# Baseline first so a regression is obvious in the scored table.
VARIANTS=(
  "legacy|penalty.target_ratio=null penalty.aggregator=null"
  "ratio001|penalty.target_ratio=0.01 penalty.aggregator=null"
  "ratio005|penalty.target_ratio=0.05 penalty.aggregator=null"
  "ratio020|penalty.target_ratio=0.20 penalty.aggregator=null"
  "relobralo|penalty.aggregator=relobralo"
  "softadapt|penalty.aggregator=soft_adapt"
)

ENTRY="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${ENTRY%%|*}"
OVERRIDES="${ENTRY#*|}"

STEPS="${STEPS:-3000}"
ACCEL="${ACCEL:-cpu}"

echo "=================================================="
echo "  Job ID    : $SLURM_JOB_ID  (task $SLURM_ARRAY_TASK_ID)"
echo "  Node      : $SLURM_NODELIST"
echo "  GPU       : $CUDA_VISIBLE_DEVICES"
echo "  Variant   : $LABEL"
echo "  Overrides : $OVERRIDES"
echo "  Steps     : $STEPS"
echo "  Accelerator: $ACCEL"
echo "  Start     : $(date)"
echo "=================================================="

# --- 3. Environment ---------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torchdisorder
CONDA_PYTHON="/home/conrard/.conda/envs/torchdisorder/bin/python"

$CONDA_PYTHON -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# The order-parameter backends have separate neighbour-list implementations.
# Confirm they agree here rather than discovering a divergence in the results:
# a wrong neighbour list changes coordination silently and invalidates the scan.
$CONDA_PYTHON -m pytest -q tests/test_fis.py -k "neighbor_list or warp_backend" 2>&1 | tail -3

export WANDB_MODE=offline

# Runs on CPU by default (ACCEL=cpu).  On an A30 this system asks for 10.81 GiB
# on top of 12.80 GiB already held and dies at step 0.  Profiling shows the
# scattering forward+backward peaks at only 0.73 GiB, so the allocation is
# elsewhere in the training loop and is NOT yet isolated -- setting
# scattering.chunk_size did not change the request by a byte.  Until that is
# understood, CPU is the honest choice: 503 GB of RAM and no silent truncation.
# Override with:  sbatch --gres=gpu:1 --export=ALL,ACCEL=cuda ...
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"

source "$PROJECT_ROOT/scripts/slurm_utils.sh"
log_hardware_info "SiO2 balance scan: $LABEL" "$CONDA_PYTHON" logs

# --- 4. Run -----------------------------------------------------------------
# constraint_warmup_steps=0: the 500-step default silently exceeds short runs,
# leaving the constraints asleep for the whole job.
$CONDA_PYTHON scripts/train.py \
    experiment_name="SiO2_bal_${LABEL}" \
    data=SiO2 \
    structure=silica \
    target=F_Q \
    max_steps="$STEPS" \
    accelerator="$ACCEL" \
    stability.constraint_warmup_steps=0 \
    +health.check_interval=250 \
    +health.expected_cn=4.0 \
    output.plot_interval=500 \
    scattering.chunk_size=20000 \
    constraints.overlap_repulsion.chunk_size=1000 \
    $OVERRIDES

log_runtime
echo "=================================================="
echo "  Done: $(date)  variant=$LABEL"
echo "=================================================="
