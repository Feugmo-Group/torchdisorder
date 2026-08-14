#!/bin/bash
# =============================================================================
# TorchDisorder — compare the routes from crystal to amorphous
# =============================================================================
#
# Four tasks: {SiO2, GeO2} x {MACE melt-quench, WWW bond switching}.
# All start from the same crystal and are scored against the same published model
# on quantities neither route fits.
#
# The question is not which gets a lower score. It is which produces a network
# whose ring topology has actually moved off the crystalline value while
# coordination survives -- the one thing displacement of atoms can never do.
# scripts/compare_to_literature.py and compare_order_params.py decide that.
#
#   sbatch scripts/slurm_glass_routes.sh
# =============================================================================
#SBATCH --job-name=routes
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/routes_%A_%a.out
#SBATCH --error=logs/routes_%A_%a.err

set -e
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
PROJECT_ROOT="/home/conrard/torchdisorder"
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT"
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
cd "$PROJECT_ROOT"; mkdir -p logs

# label | system args | route
TASKS=(
  "SiO2_mq|Si O 2.2 4 2.20 c-SiO2|meltquench"
  "GeO2_mq|Ge O 2.4 4 3.65 c-GeO2|meltquench"
  "SiO2_www|Si O 2.2 4 2.20 sio2_from_crystal|www"
  "GeO2_www|Ge O 2.4 4 3.65 geo2_from_crystal|www"
)
ENTRY="${TASKS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${ENTRY%%|*}"; REST="${ENTRY#*|}"
SPEC="${REST%%|*}"; ROUTE="${REST#*|}"
read C N CUT CN RHO SRC <<< "$SPEC"

echo "=== $LABEL  route=$ROUTE  $C-$N cutoff=$CUT rho=$RHO  $(date) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torchdisorder
PY="/home/conrard/.conda/envs/torchdisorder/bin/python"
$PY -c "import torch; print('CUDA:', torch.cuda.is_available())"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="data/crystal-structures/${LABEL}.cif"

if [ "$ROUTE" = "meltquench" ]; then
  # MACE on ~1125 atoms fits an A30 easily; the earlier OOM was the 5184-atom
  # scattering calculation, not the potential.
  $PY scripts/build_glass_melt_quench.py \
      --input "data/crystal-structures/${SRC}.cif" --output "$OUT" \
      --central "$C" --neighbour "$N" --cutoff "$CUT" --expected-cn "$CN" \
      --density "$RHO" --device cuda \
      --melt-steps "${MELT:-20000}" --quench-steps "${QUENCH:-30000}" \
      --anneal-steps "${ANNEAL:-5000}" --log-every 2000
else
  # Relaxation is not optional here: an unrelaxed transposition reads as
  # <CN> = 3.56 at the first-shell cutoff purely because the new bonds are
  # stretched, recovering to 4.03 by 2.4 A.
  $PY scripts/run_bond_switch.py \
      --input "data/crystal-structures/${SRC}.cif" --output "$OUT" \
      --central "$C" --neighbour "$N" --cutoff "$CUT" --expected-cn "$CN" \
      --steps "${SWITCHES:-3000}" --device cuda --mlip --relax-steps 30 --log-every 100
fi

case "$C" in
  Si) REF="data/json/sio2_glass_gap.cif"; SYS=SiO2 ;;
  Ge) REF="data/json/geo2_glass_nnp.cif"; SYS=GeO2 ;;
esac
if [ -f "$OUT" ]; then
  echo; echo "===== scored against the published model ====="
  $PY scripts/compare_to_literature.py --test "$OUT" --reference "$REF" \
      --system "$SYS" --label "$LABEL" 2>/dev/null || true
  $PY scripts/compare_order_params.py --reference "$REF" --test "$OUT" \
      --labels "$LABEL" --central "$C" --neighbour "$N" --cutoff "$CUT" 2>/dev/null || true
fi
echo "=== done $LABEL $(date) ==="
