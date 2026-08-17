#!/bin/bash
# =============================================================================
# TorchDisorder — re-melt GeO2 at a temperature GeO2 can actually survive
# =============================================================================
#
# GeO2_mq.cif is contaminated with molecular oxygen. Eleven O-O pairs sit at
# 1.238-1.242 A, essentially the gas-phase O2 bond length of 1.21 A, and they
# are what sets the structure's min-distance to 1.238 A. The knock-on is visible
# in the coordination: <CN> = 3.960 with Ge atoms stranded at CN = 1, 2 and 3,
# because every O2 that forms removes two oxygens from the network.
#
# The cause is the melt temperature, and it was inherited by accident.
# slurm_glass_routes.sh never passes --melt-temp, so GeO2 took the script's
# default of 4000 K -- a value chosen for silica. SiO2 melts at 1986 K, so 4000 K
# is 2.0x its melting point and merely a hot liquid. GeO2 melts at 1388 K, so the
# same 4000 K is 2.9x, far into the range where GeO2 thermally reduces:
#
#     GeO2  ->  GeO + 1/2 O2
#
# That is not a simulation artefact to be tuned away, it is the potential
# correctly describing decomposition of a real material held far too hot.
#
# This script melts at 2800 K instead, which reproduces silica's 2.0x ratio
# against GeO2's own melting point. Everything else matches slurm_sio2_remelt.sh,
# including gamma = 1.0 ps^-1 -- at the 0.1 default the thermostat relaxes over
# 10 ps against a 30 ps melt and never delivers its setpoint (see
# slurm_sio2_remelt.sh, and the SiO2 run that motivated it).
#
# Note the two systems fail in OPPOSITE directions and must not be "fixed" the
# same way: silica needed a melt that got hot enough to lose its topology,
# germania needs one that stays cool enough to keep its oxygen.
#
# The number to judge this by is the O2 count (must be 0) and q4 against the
# published NNP reference -- NOT the built-in PASS verdict, which a structure
# full of O2 can still earn since min-distance and mean CN both stay plausible.
#
#   sbatch scripts/slurm_geo2_remelt.sh
#   sbatch --export=ALL,MELT_T=2500,TAG=_cool scripts/slurm_geo2_remelt.sh
# =============================================================================
#SBATCH --job-name=geo2remelt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/geo2remelt_%j.out
#SBATCH --error=logs/geo2remelt_%j.err

set -e
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
PROJECT_ROOT="/home/conrard/torchdisorder"
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT"
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
cd "$PROJECT_ROOT"; mkdir -p logs

# Triton JIT-compiles a CUDA shim at first use and picks the first compiler on
# PATH, which on yemba is the NVIDIA HPC SDK's nvc. nvc rejects the gcc flags
# Triton emits ("nvc-Error-Unknown switch: -Wno-psabi") and the job dies seconds
# in. Force gcc. Harmless for MACE, required for MatterSim.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

# One env per potential. mace-torch pins e3nn==0.4.4 while MatterSim and
# SevenNet need e3nn>=0.5, and the pin is real -- forcing MACE onto e3nn 0.6.0
# loads the model and then dies computing forces on the changed Irreps API.
POTENTIAL="${POTENTIAL:-mace}"
case "$POTENTIAL" in
  mace)      CONDA_ENV=torchdisorder; DEF_MODEL=medium-mpa-0 ;;
  mattersim) CONDA_ENV=mlip;          DEF_MODEL=MatterSim-v1.0.0-5M.pth ;;
  sevennet)  CONDA_ENV=sevennet;      DEF_MODEL=7net-mf-ompa ;;
  *) echo "unknown POTENTIAL=$POTENTIAL" >&2; exit 1 ;;
esac

MELT_T="${MELT_T:-2800}"
echo "=== GeO2 re-melt (potential=$POTENTIAL melt_T=$MELT_T gamma=${GAMMA:-1.0} melt=${MELT:-30000}) $(date) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
PY="/home/conrard/.conda/envs/$CONDA_ENV/bin/python -u"
$PY -c "import torch; print('CUDA:', torch.cuda.is_available())"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TAG keeps a second arm (e.g. a different melt temperature) from overwriting
# the first, so the two can be compared rather than one silently replacing.
OUT="data/crystal-structures/GeO2_mq_hot${TAG:-}.cif"

# rho = 3.65 g/cm3 and cutoff 2.4 A carried over from slurm_glass_routes.sh,
# where they were correct -- the melt temperature was the only wrong parameter.
$PY scripts/build_glass_melt_quench.py \
    --input data/crystal-structures/c-GeO2.cif --output "$OUT" \
    --central Ge --neighbour O --cutoff 2.4 --expected-cn 4 \
    --density 3.65 --device cuda \
    --gamma "${GAMMA:-1.0}" --potential "$POTENTIAL" --model "${MODEL:-$DEF_MODEL}" \
    --melt-temp "$MELT_T" --melt-steps "${MELT:-30000}" \
    --quench-steps "${QUENCH:-30000}" --anneal-steps "${ANNEAL:-5000}" \
    --log-every 2000

REF="data/json/geo2_glass_nnp.cif"
if [ -f "$OUT" ]; then
  # The check the built-in verdict does not make. O2 is invisible to a mean-CN
  # test and to a min-distance floor set for Ge-O, so count it explicitly.
  echo; echo "===== molecular O2 check ====="
  $PY -c "
from ase.io import read
a = read('$OUT'); sym = a.get_chemical_symbols(); d = a.get_all_distances(mic=True)
O = [i for i, s in enumerate(sym) if s == 'O']
o2 = [(i, j) for k, i in enumerate(O) for j in O[k+1:] if d[i, j] < 1.35]
print(f'O2 molecules (O-O < 1.35 A): {len(o2)}')
for i, j in o2[:10]:
    print(f'    O{i}-O{j}  {d[i, j]:.3f} A')
print('CLEAN' if not o2 else 'CONTAMINATED -- melt is still too hot')
raise SystemExit(0 if not o2 else 1)
" || true

  echo; echo "===== scored against the published model ====="
  $PY scripts/compare_to_literature.py --test "$OUT" --reference "$REF" \
      --system GeO2 --label "GeO2_mq_hot${TAG:-}" 2>/dev/null || true
  $PY scripts/compare_order_params.py --reference "$REF" --test "$OUT" \
      --labels "GeO2_mq_hot${TAG:-}" --central Ge --neighbour O --cutoff 2.4 2>/dev/null || true
fi
echo "=== done $(date) ==="
