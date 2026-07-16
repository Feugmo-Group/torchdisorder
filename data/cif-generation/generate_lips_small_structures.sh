#!/usr/bin/env bash
# =============================================================================
# Generate all 6 small LiPS glass structures (noLi + withLi × 67/70/75%)
#
# Supercells (small vs full):
#   67% / 70%  Li7P3S11  :  4×2×2  near-cubic ~20 Å  (vs 5×8×5 full)
#   75%        Li3PS4_β  :  2×2×3  near-cubic ~16 Å  (vs 5×6×9 full)
#
# Outputs (in this directory):
#   glass_{67,70,75}Li2S_small_{noLi,withLi}.cif
#   glass_{67,70,75}Li2S_small_{noLi,withLi}_constraints.json
#
# Then copies to:
#   ../crystal-structures/   (CIF files)
#   ../json/                 (constraint JSON files)
#
# Usage:
#   cd data/cif-generation
#   bash generate_lips_small_structures.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(which python)}"
SEED=42

echo ""
echo "============================================================"
echo "  TorchDisorder — Small LiPS Structure Generation"
echo "  Python : $PYTHON"
echo "  Seed   : $SEED"
echo "============================================================"
echo ""

# ── 67Li2S-33P2S5  (Li7P3S11, 4×2×2) ────────────────────────────────────────
# 4×2×2 gives near-cubic ~25×25×25 Å → ~20 Å after density scaling (vs 2×4×2
# which gave a=12, b=51, c=25 Å — too elongated for a valid MIC/RDF calculation)
echo "[1/6] 67Li2S-33P2S5  noLi  (4×2×2)..."
$PYTHON lps_generator.py \
    --input Li7P3S11.cif \
    --target 67Li2S-33P2S5 \
    --supercell 4,2,2 \
    --output glass_67Li2S_small_noLi \
    --seed $SEED

echo "[2/6] 67Li2S-33P2S5  withLi  (4×2×2)..."
$PYTHON lps_generator.py \
    --input Li7P3S11.cif \
    --target 67Li2S-33P2S5 \
    --supercell 4,2,2 \
    --output glass_67Li2S_small_withLi \
    --seed $SEED \
    --keep-li

# ── 70Li2S-30P2S5  (Li7P3S11, 4×2×2) ────────────────────────────────────────
echo "[3/6] 70Li2S-30P2S5  noLi  (4×2×2)..."
$PYTHON lps_generator.py \
    --input Li7P3S11.cif \
    --target 70Li2S-30P2S5 \
    --supercell 4,2,2 \
    --output glass_70Li2S_small_noLi \
    --seed $SEED

echo "[4/6] 70Li2S-30P2S5  withLi  (4×2×2)..."
$PYTHON lps_generator.py \
    --input Li7P3S11.cif \
    --target 70Li2S-30P2S5 \
    --supercell 4,2,2 \
    --output glass_70Li2S_small_withLi \
    --seed $SEED \
    --keep-li

# ── 75Li2S-25P2S5  (Li3PS4_beta, 2×2×3) ─────────────────────────────────────
echo "[5/6] 75Li2S-25P2S5  noLi  (2×2×3)..."
$PYTHON lps_generator.py \
    --input Li3PS4_beta.cif \
    --target 75Li2S-25P2S5 \
    --supercell 2,2,3 \
    --output glass_75Li2S_small_noLi \
    --seed $SEED

echo "[6/6] 75Li2S-25P2S5  withLi  (2×2×3)..."
$PYTHON lps_generator.py \
    --input Li3PS4_beta.cif \
    --target 75Li2S-25P2S5 \
    --supercell 2,2,3 \
    --output glass_75Li2S_small_withLi \
    --seed $SEED \
    --keep-li

# ── Copy outputs ──────────────────────────────────────────────────────────────
echo ""
echo "Copying CIF files → ../crystal-structures/"
cp glass_*small_noLi.cif glass_*small_withLi.cif ../crystal-structures/

echo "Copying JSON files → ../json/"
cp glass_*small_noLi_constraints.json glass_*small_withLi_constraints.json ../json/

echo ""
echo "============================================================"
echo "  All small structures generated successfully."
echo ""
echo "  CIF files → data/crystal-structures/"
ls ../crystal-structures/glass_*small*.cif 2>/dev/null | sed 's/.*\//    /'
echo ""
echo "  JSON files → data/json/"
ls ../json/glass_*small*_constraints.json 2>/dev/null | sed 's/.*\//    /'
echo "============================================================"
echo ""
