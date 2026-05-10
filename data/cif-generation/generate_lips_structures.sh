#!/usr/bin/env bash
# Generate LiPS glass structures for all three compositions, with and without Li.
#
# Usage:
#   ./generate_lips_structures.sh [--seed N] [--disorder F] [--skip-no-li] [--skip-with-li]
#
# Generated CIF files are copied to:   ../crystal-structures/
# Generated constraint JSONs go to:    ../json/
#
# Output naming convention:
#   glass_{67,70,75}Li2S_{noLi,withLi}.cif
#   glass_{67,70,75}Li2S_{noLi,withLi}_constraints.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="${SCRIPT_DIR}/lps_generator.py"
CIF_DIR="${SCRIPT_DIR}/../crystal-structures"
JSON_DIR="${SCRIPT_DIR}/../json"

# Defaults
SEED=42
DISORDER=0.3
RUN_NO_LI=true
RUN_WITH_LI=true

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)          SEED="$2";     shift 2 ;;
        --disorder)      DISORDER="$2"; shift 2 ;;
        --skip-no-li)    RUN_NO_LI=false;   shift ;;
        --skip-with-li)  RUN_WITH_LI=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$SCRIPT_DIR"

echo "=========================================================================="
echo "  LiPS Glass Structure Generator — all 3 compositions × {noLi, withLi}"
echo "  seed=${SEED}  disorder=${DISORDER}"
echo "  CIFs  -> ${CIF_DIR}"
echo "  JSONs -> ${JSON_DIR}"
echo "=========================================================================="
echo ""

# --------------------------------------------------------------------------
# Helper — run one variant, then distribute output files
# --------------------------------------------------------------------------
run_variant() {
    local label="$1"
    local input="$2"
    local target="$3"
    local supercell="$4"
    local output="$5"     # local prefix (stays in cif-generation/)
    local extra="${6:-}"  # optional: --keep-li

    echo "------------------------------------------------------------------"
    echo "  ${label}"
    echo "------------------------------------------------------------------"
    python "${GEN}" \
        --input     "${input}" \
        --target    "${target}" \
        --supercell "${supercell}" \
        --output    "${output}" \
        --disorder  "${DISORDER}" \
        --seed      "${SEED}" \
        ${extra}

    # Copy CIF to crystal-structures
    if [[ -f "${output}.cif" ]]; then
        cp "${output}.cif" "${CIF_DIR}/${output}.cif"
        echo "  Copied ${output}.cif -> crystal-structures/"
    fi

    # Copy constraints JSON to data/json
    if [[ -f "${output}_constraints.json" ]]; then
        cp "${output}_constraints.json" "${JSON_DIR}/${output}_constraints.json"
        echo "  Copied ${output}_constraints.json -> json/"
    fi
    echo ""
}

# --------------------------------------------------------------------------
# 67Li2S–33P2S5   (supercell 5×8×5 of Li7P3S11)
# --------------------------------------------------------------------------
if $RUN_NO_LI; then
    run_variant \
        "67Li2S-33P2S5 — no Li" \
        "Li7P3S11.cif" \
        "67Li2S-33P2S5" \
        "5,8,5" \
        "glass_67Li2S_noLi"
fi

if $RUN_WITH_LI; then
    run_variant \
        "67Li2S-33P2S5 — with Li" \
        "Li7P3S11.cif" \
        "67Li2S-33P2S5" \
        "5,8,5" \
        "glass_67Li2S_withLi" \
        "--keep-li"
fi

# --------------------------------------------------------------------------
# 70Li2S–30P2S5   (supercell 5×8×5 of Li7P3S11)
# --------------------------------------------------------------------------
if $RUN_NO_LI; then
    run_variant \
        "70Li2S-30P2S5 — no Li" \
        "Li7P3S11.cif" \
        "70Li2S-30P2S5" \
        "5,8,5" \
        "glass_70Li2S_noLi"
fi

if $RUN_WITH_LI; then
    run_variant \
        "70Li2S-30P2S5 — with Li" \
        "Li7P3S11.cif" \
        "70Li2S-30P2S5" \
        "5,8,5" \
        "glass_70Li2S_withLi" \
        "--keep-li"
fi

# --------------------------------------------------------------------------
# 75Li2S–25P2S5   (supercell 5×6×9 of Li3PS4_beta)
# --------------------------------------------------------------------------
if $RUN_NO_LI; then
    run_variant \
        "75Li2S-25P2S5 — no Li" \
        "Li3PS4_beta.cif" \
        "75Li2S-25P2S5" \
        "5,6,9" \
        "glass_75Li2S_noLi"
fi

if $RUN_WITH_LI; then
    run_variant \
        "75Li2S-25P2S5 — with Li" \
        "Li3PS4_beta.cif" \
        "75Li2S-25P2S5" \
        "5,6,9" \
        "glass_75Li2S_withLi" \
        "--keep-li"
fi

echo "=========================================================================="
echo "  All structures generated successfully."
echo "  CIF files   -> ${CIF_DIR}/"
echo "  JSON files  -> ${JSON_DIR}/"
echo "=========================================================================="
