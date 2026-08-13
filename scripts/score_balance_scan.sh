#!/usr/bin/env bash
# Score every variant of the balancing scan against the published reference.
#
# Run after slurm_sio2_balance_scan.sh finishes:
#     bash scripts/score_balance_scan.sh
#
# Prints one comparison table per variant.  The quantities that decide it are the
# ones the refinement never fitted: bond-length spread, O-Si-O and Si-O-Si
# angles, and the second-neighbour shells.  A variant that fits F(Q) well but
# drifts on these is overfitting a one-dimensional projection.
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
REFERENCE="data/json/sio2_glass_gap.cif"

if [[ ! -f "$REFERENCE" ]]; then
    echo "Reference structure missing: $REFERENCE" >&2
    exit 1
fi

shopt -s nullglob
found=0
for label in legacy ratio001 ratio005 ratio020 relobralo softadapt; do
    # Most recent run for this variant.
    latest=""
    for d in outputs/SiO2_bal_${label}_*/*/final_results/final_structure.cif; do
        latest="$d"
    done
    if [[ -z "$latest" ]]; then
        echo "--- ${label}: no completed run found"
        continue
    fi
    found=$((found + 1))
    echo
    echo "############ ${label} ############"
    health="$(dirname "$latest")/structure_health.txt"
    [[ -f "$health" ]] && cat "$health"
    "$PYTHON" scripts/compare_to_literature.py \
        --test "$latest" --reference "$REFERENCE" \
        --system SiO2 --label "$label" 2>/dev/null
done

if [[ $found -eq 0 ]]; then
    echo "No completed runs found under outputs/SiO2_bal_*/" >&2
    exit 1
fi

echo
echo "Pick by the unfitted quantities, not by chi^2:"
echo "  * Si-O spread near the reference 0.033 A (the smearing we are chasing)"
echo "  * Si-O-Si angle inside 142-147 deg"
echo "  * O-O and Si-Si shells resolved and at 2.63 / 3.08 A"
echo "A variant winning on chi^2 while losing these is fitting the curve, not the structure."
