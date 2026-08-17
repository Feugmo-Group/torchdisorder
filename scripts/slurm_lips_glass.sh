#!/bin/bash
# =============================================================================
# TorchDisorder — regenerate the Li-P-S glasses by melt-quench
# =============================================================================
#
# Why this exists: every glass_*Li2S*.cif in the repo contains overlapping atoms,
# down to 0.069 A between Li and S where the bond is 2.4 A. The cause is in
# lps_generator.py, which pairs P4 sites arbitrarily rather than by proximity and
# then drags them together across whatever lies between, with no overlap check
# and with atom indices that go stale after each removal. Patching that is not
# worth it: melt-quench reaches the same compositions through a path that cannot
# produce an overlap, because MACE will not tolerate one.
#
# Same code path as the SiO2/GeO2 routes, which is the point -- one generator for
# all three systems.
#
# Melt temperature is 1500 K, NOT the 4000 K used for silica. Li2S-P2S5 melts
# near 900-1100 K, so 1500 K gives a well-mixed liquid with margin while keeping
# P-S bonds intact; 4000 K would dissociate the sulfide and destroy the very P
# environment distribution these structures exist to represent.
#
# gamma is 1.0 ps^-1, not the 0.1 default. At 0.1 the thermostat relaxes over
# 10 ps against a 20 ps melt: the first attempt (job 1207) initialised at 1420 K,
# collapsed to ~850 K by 2 ps as equipartition took half the kinetic energy, and
# had only recovered to ~1300 K by the end -- so it sat at or BELOW its own
# melting point throughout and never became a liquid. That, not MACE's P-S
# accuracy, is why the network failed to finish connecting: <CN> came out 3.615
# for Li4P2S7 and 3.802 for Li7P3S11, worst where the composition needs the most
# bridging sulfur.
#
#   sbatch scripts/slurm_lips_glass.sh
# =============================================================================
#SBATCH --job-name=lipsglass
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lipsglass_%A_%a.out
#SBATCH --error=logs/lipsglass_%A_%a.err

set -e
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
PROJECT_ROOT="/home/conrard/torchdisorder"
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT"
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
cd "$PROJECT_ROOT"; mkdir -p logs

# Triton JIT-compiles a CUDA shim on first use and takes the first compiler on
# PATH, which on yemba is the NVIDIA HPC SDK's nvc. nvc rejects the gcc flags
# Triton emits ("nvc-Error-Unknown switch: -Wno-psabi") and the job dies seconds
# in, after the scheduler has already handed out a GPU. Harmless for MACE.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

# 16G rather than the 80G used previously: the node has 250 GB total and one
# over-reserved job blocks every other for its full walltime. Measured use of a
# ~1000-atom MACE melt-quench is a few GB.
TASKS=(
  "lips67|Li4P2S7   67Li2S-33P2S5"
  "lips70|Li7P3S11  70Li2S-30P2S5"
  "lips75|Li3PS4    75Li2S-25P2S5"
)
ENTRY="${TASKS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${ENTRY%%|*}"; DESC="${ENTRY#*|}"

# One env per potential. mace-torch pins e3nn==0.4.4 while MatterSim and
# SevenNet need e3nn>=0.5, and the pin is real -- forcing MACE onto e3nn 0.6.0
# loads the model and then dies computing forces on the changed Irreps API.
# MatterSim and SevenNet *could* share, but sharing already cost us once:
# installing SevenNet downgraded nvidia-nccl-cu13 to 2.29.7 and left MatterSim's
# torch unimportable with "undefined symbol: ncclCommResume". One env each.
POTENTIAL="${POTENTIAL:-mace}"
# LiPS-25 (Fragapane & Deringer, JCTC 2026) is a MACE model trained from scratch
# on the Li2S-P2S5 tie-line, covering all three compositions here plus 1750
# melt-quench cells generated specifically to be amorphous. It exists in this
# list because every universal potential we have tried reduces P(V) to P(IV) in
# the melt. Default is the paper's recommended 6 A cutoff, seed 1; the other
# four seeds sit beside it and are the cheapest uncertainty estimate available.
# Use the from-scratch models, NOT models/fine-tuned/ -- the paper's own result
# is that fine-tuned foundation models were WORSE than zero-shot in the liquid.
LIPS25=/data/scratch/conrard/lips25/lips-25/models/mace-models
case "$POTENTIAL" in
  mace)      CONDA_ENV=torchdisorder; DEF_MODEL=medium-mpa-0 ;;
  mattersim) CONDA_ENV=mlip;          DEF_MODEL=MatterSim-v1.0.0-5M.pth ;;
  sevennet)  CONDA_ENV=sevennet;      DEF_MODEL=7net-mf-ompa ;;
  graphpes)  CONDA_ENV=graphpes;      DEF_MODEL="$LIPS25/cutoff/model_mace_cutoff_6_1.pt" ;;
  *) echo "unknown POTENTIAL=$POTENTIAL" >&2; exit 1 ;;
esac

echo "=== $LABEL  ($DESC)  potential=$POTENTIAL env=$CONDA_ENV  $(date) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
# -u: batch stdout is a file, so Python block-buffers it and a long run looks
# hung until it exits.
PY="/home/conrard/.conda/envs/$CONDA_ENV/bin/python -u"
$PY -c "import torch; print('CUDA:', torch.cuda.is_available())"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="data/crystal-structures/${LABEL}_from_crystal.cif"
# TAG keeps a second arm (e.g. a different foundation model) from overwriting
# the first, so the two can be compared rather than one silently replacing.
OUT="data/crystal-structures/${LABEL}_glass${TAG:-}.cif"

$PY scripts/build_glass_melt_quench.py \
    --input "$SEED" --output "$OUT" \
    --central P --neighbour S --cutoff 2.5 --expected-cn 4 \
    --density 1.85 --device cuda \
    --gamma "${GAMMA:-1.0}" --potential "$POTENTIAL" --model "${MODEL:-$DEF_MODEL}" \
    --melt-temp "${MELT_T:-1500}" --quench-temp 300 \
    --melt-steps "${MELT:-30000}" --quench-steps "${QUENCH:-40000}" \
    --anneal-steps "${ANNEAL:-5000}" --log-every 2000

# The check that matters: zero pairs inside the covalent floor. The structures
# this replaces would all have failed it.
if [ -f "$OUT" ]; then
  echo; echo "===== overlap check ====="
  $PY -c "
from ase.io import read
from torchdisorder.common.validation import validate_structure
rep = validate_structure(read('$OUT'), check_plateau=True, central='P',
                         neighbour='S', bond_cutoff=2.5, expected_cn=4)
print(rep.summary())
raise SystemExit(0 if rep else 1)
"
fi
echo "=== done $LABEL $(date) ==="
