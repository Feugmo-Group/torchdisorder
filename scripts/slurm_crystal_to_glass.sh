#!/bin/bash
# =============================================================================
# TorchDisorder — crystal -> amorphous, scored against published models
# =============================================================================
#
# Starts from a CRYSTAL (expanded to glass density, 0.02 A symmetry-breaking rattle)
# and asks whether the refinement can FIND a glass, driven by the scattering data and
# the local-environment constraints. That is a harder and more informative question
# than whether it preserves a glass it was handed.
#
# Local order-parameter targets are measured on the published glass for each system,
# so the constraints describe the tetrahedra we want; the network topology is left
# entirely to the data. Success means the O-Si-O spread widens from ~1 to ~5 deg and
# Si-Si/Si moves off the crystalline 4.0 -- WITHOUT coordination collapsing.
#
# READ THIS BEFORE TRUSTING THE OUTPUT.  Forwarding either reference structure
# through the model gives a calculated F(Q) roughly 3x larger than the measured
# one, even at the calibrated kernel_width, and the discrepancy reproduces across
# both materials, two laboratories and two Ge isotopes -- so it is the forward
# model, not the data.  chi^2 is therefore dominated by that mismatch rather than
# by structural error, and driving it down distorts the structure.  Until it is
# resolved these runs measure how far the refinement drags a good structure away
# from itself; they do not demonstrate that it finds a better one.  Judge them
# with scripts/compare_to_literature.py and scripts/compare_order_params.py, never
# by chi^2 alone.
#
# Submit:
#     sbatch scripts/slurm_long_runs.sh                  # 5000 steps, both systems
#     sbatch --export=ALL,STEPS=20000 scripts/slurm_long_runs.sh
# =============================================================================
#SBATCH --job-name=tdcryst
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cryst_%A_%a.out
#SBATCH --error=logs/cryst_%A_%a.err

set -e

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
PROJECT_ROOT="/home/conrard/torchdisorder"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD

cd "$PROJECT_ROOT"
mkdir -p logs

SYSTEMS=(
  "SiO2|data=SiO2 structure=silica_from_crystal data.json_path=$PROJECT_ROOT/data/json/sio2_from_crystal_constraints.json"
  "GeO2|data=GeO2 structure=germania_from_crystal data.json_path=$PROJECT_ROOT/data/json/geo2_from_crystal_constraints.json"
)
ENTRY="${SYSTEMS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${ENTRY%%|*}"
OVERRIDES="${ENTRY#*|}"

STEPS="${STEPS:-5000}"
ACCEL="${ACCEL:-cpu}"

echo "=================================================="
echo "  Job       : $SLURM_JOB_ID (task $SLURM_ARRAY_TASK_ID)"
echo "  System    : $LABEL"
echo "  Steps     : $STEPS   Accelerator: $ACCEL"
echo "  Start     : $(date)"
echo "=================================================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torchdisorder
CONDA_PYTHON="/home/conrard/.conda/envs/torchdisorder/bin/python"

# The neighbour list must agree with ASE on a non-orthogonal cell before any of
# this means anything: a wrong one changes coordination silently.
$CONDA_PYTHON -m pytest -q tests/test_fis.py -k "neighbor_list or warp_backend" 2>&1 | tail -2

export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$PROJECT_ROOT/scripts/slurm_utils.sh"
log_hardware_info "long run: $LABEL" "$CONDA_PYTHON" logs

# constraint_warmup_steps stays at its default here: with 5000+ steps the 500-step
# warmup is a genuine warmup rather than the whole run.
$CONDA_PYTHON scripts/train.py \
    experiment_name="${LABEL}_from_crystal" \
    target=F_Q \
    max_steps="$STEPS" \
    accelerator="$ACCEL" \
    +health.check_interval=500 \
    +health.expected_cn=4.0 \
    output.plot_interval=1000 \
    output.trajectory_interval=1000 \
    checkpoint_interval=1000 \
    scattering.chunk_size=20000 \
    constraints.overlap_repulsion.chunk_size=1000 \
    $OVERRIDES

log_runtime

# Score against the reference on quantities the refinement never fitted.
case "$LABEL" in
  SiO2) REF="data/json/sio2_glass_gap.cif"; C=Si; N=O; CUT=2.2 ;;
  GeO2) REF="data/json/geo2_glass_nnp.cif"; C=Ge; N=O; CUT=2.4 ;;
esac
FINAL=$(ls -d outputs/${LABEL}_from_crystal_*/*/final_results/final_structure.cif 2>/dev/null | tail -1)
if [ -n "$FINAL" ]; then
  echo; echo "===== independent scoring ====="
  $CONDA_PYTHON scripts/compare_to_literature.py --test "$FINAL" --reference "$REF" \
      --system "$LABEL" --label "${LABEL}_from_crystal" 2>/dev/null || true
  $CONDA_PYTHON scripts/compare_order_params.py --reference "$REF" --test "$FINAL" \
      --labels "${LABEL}_from_crystal" --central "$C" --neighbour "$N" --cutoff "$CUT" 2>/dev/null || true
fi

echo "=================================================="
echo "  Done: $(date)  $LABEL"
echo "=================================================="
