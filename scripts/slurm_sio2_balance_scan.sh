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
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
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

echo "=================================================="
echo "  Job ID    : $SLURM_JOB_ID  (task $SLURM_ARRAY_TASK_ID)"
echo "  Node      : $SLURM_NODELIST"
echo "  GPU       : $CUDA_VISIBLE_DEVICES"
echo "  Variant   : $LABEL"
echo "  Overrides : $OVERRIDES"
echo "  Steps     : $STEPS"
echo "  Start     : $(date)"
echo "=================================================="

# --- 3. Environment ---------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torchdisorder
CONDA_PYTHON="/home/conrard/.conda/envs/torchdisorder/bin/python"

$CONDA_PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# The order-parameter backends have separate neighbour-list implementations.
# Confirm they agree here rather than discovering a divergence in the results:
# a wrong neighbour list changes coordination silently and invalidates the scan.
$CONDA_PYTHON -m pytest -q tests/test_fis.py -k "neighbor_list or warp_backend" 2>&1 | tail -3

export WANDB_MODE=offline

# Memory: 5184 atoms means 26.9M pairs through O(N^2) kernels.  The chunk-size
# autosizer reads *free* GPU memory once at startup, which overshoots whenever
# another job lands on the same card -- that is what produced a 10.8 GiB
# allocation request against 10.5 GiB free and killed the first attempt at step 0.
# Size them explicitly instead of letting the autosizer guess.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    accelerator=cuda \
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
