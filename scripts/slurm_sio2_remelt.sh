#!/bin/bash
# =============================================================================
# TorchDisorder — re-melt SiO2 with a thermostat that actually reaches 4000 K
# =============================================================================
#
# The first attempt (job 1203_0) produced a network that was still partly
# crystalline: q4 = 0.2970 against the published GAP glass's 0.1416, a shift of
# +0.155 that exceeds the reference's own sigma of 0.095. The tetrahedra came out
# too regular (O-Si-O spread 3.77 deg vs the real glass's 5.38) and the network
# too open (Si-O-Si 151.7 deg vs 140.8). GeO2 under identical settings matched
# its reference on every order parameter, so this is specific to silica.
#
# The cause is thermostat coupling, not the method. Langevin damping defaults to
# 0.1 ps^-1, i.e. a 10 ps relaxation time, against a 20 ps melt. Velocities are
# initialised at 4000 K, equipartition immediately halves the kinetic temperature
# to ~2690 K, and the thermostat needs ~10 ps to pull it back -- the run only
# reached ~3730 K at the very END of the melt, so silica never spent meaningful
# time hot enough to lose its network topology.
#
# Two changes: gamma 0.1 -> 1.0 ps^-1 (1 ps relaxation, so the setpoint is
# reached within ~2-3 ps) and a 30 ps rather than 20 ps melt. That buys ~27 ps
# genuinely at 4000 K instead of ~0.
#
# The number to judge this by is q4 against the reference, NOT the built-in PASS
# verdict -- the melt-quench script says so itself, since a hot crystal passes.
#
#   sbatch scripts/slurm_sio2_remelt.sh
# =============================================================================
#SBATCH --job-name=sio2remelt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sio2remelt_%j.out
#SBATCH --error=logs/sio2remelt_%j.err

set -e
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
PROJECT_ROOT="/home/conrard/torchdisorder"
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT"
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
cd "$PROJECT_ROOT"; mkdir -p logs

# One env per potential. mace-torch pins e3nn==0.4.4 while MatterSim and
# SevenNet need e3nn>=0.5, and the pin is real -- forcing MACE onto e3nn 0.6.0
# loads the model and then dies computing forces on the changed Irreps API.
# MatterSim and SevenNet *could* share, but sharing already cost us once:
# installing SevenNet downgraded nvidia-nccl-cu13 to 2.29.7 and left MatterSim's
# torch unimportable with "undefined symbol: ncclCommResume". One env each.
POTENTIAL="${POTENTIAL:-mace}"
case "$POTENTIAL" in
  mace)      CONDA_ENV=torchdisorder; DEF_MODEL=medium-mpa-0 ;;
  mattersim) CONDA_ENV=mlip;          DEF_MODEL=MatterSim-v1.0.0-5M.pth ;;
  sevennet)  CONDA_ENV=sevennet;      DEF_MODEL=7net-mf-ompa ;;
  *) echo "unknown POTENTIAL=$POTENTIAL" >&2; exit 1 ;;
esac

echo "=== SiO2 re-melt (potential=$POTENTIAL gamma=${GAMMA:-1.0} melt=${MELT:-30000}) $(date) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
PY="/home/conrard/.conda/envs/$CONDA_ENV/bin/python -u"
$PY -c "import torch; print('CUDA:', torch.cuda.is_available())"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TAG keeps a second arm (e.g. a different foundation model) from overwriting
# the first, so the two can be compared rather than one silently replacing.
OUT="data/crystal-structures/SiO2_mq_hot${TAG:-}.cif"

$PY scripts/build_glass_melt_quench.py \
    --input data/crystal-structures/c-SiO2.cif --output "$OUT" \
    --central Si --neighbour O --cutoff 2.2 --expected-cn 4 \
    --density 2.20 --device cuda \
    --gamma "${GAMMA:-1.0}" --potential "$POTENTIAL" --model "${MODEL:-$DEF_MODEL}" \
    --melt-temp 4000 --melt-steps "${MELT:-30000}" \
    --quench-steps "${QUENCH:-30000}" --anneal-steps "${ANNEAL:-5000}" \
    --log-every 2000

REF="data/json/sio2_glass_gap.cif"
if [ -f "$OUT" ]; then
  echo; echo "===== scored against the published model ====="
  $PY scripts/compare_to_literature.py --test "$OUT" --reference "$REF" \
      --system SiO2 --label "SiO2_mq_hot${TAG:-}" 2>/dev/null || true
  # q4 is the discriminator: it is what caught the first attempt, while F_IS
  # moved only -0.015 and would have let it through.
  $PY scripts/compare_order_params.py --reference "$REF" --test "$OUT" \
      --labels "SiO2_mq_hot${TAG:-}" --central Si --neighbour O --cutoff 2.2 2>/dev/null || true
fi
echo "=== done $(date) ==="
