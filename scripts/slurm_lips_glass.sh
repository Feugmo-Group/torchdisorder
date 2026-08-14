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

echo "=== $LABEL  ($DESC)  $(date) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torchdisorder
# -u: batch stdout is a file, so Python block-buffers it and a long run looks
# hung until it exits.
PY="/home/conrard/.conda/envs/torchdisorder/bin/python -u"
$PY -c "import torch; print('CUDA:', torch.cuda.is_available())"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="data/crystal-structures/${LABEL}_from_crystal.cif"
OUT="data/crystal-structures/${LABEL}_glass.cif"

$PY scripts/build_glass_melt_quench.py \
    --input "$SEED" --output "$OUT" \
    --central P --neighbour S --cutoff 2.5 --expected-cn 4 \
    --density 1.85 --device cuda \
    --melt-temp "${MELT_T:-1500}" --quench-temp 300 \
    --melt-steps "${MELT:-20000}" --quench-steps "${QUENCH:-30000}" \
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
