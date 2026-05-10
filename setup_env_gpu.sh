#!/usr/bin/env bash
# =============================================================================
# TorchDisorder – Conda Environment Setup  (Linux x86_64, NVIDIA GPU)
# =============================================================================
#
# Creates a conda environment called "torchdisorder" and installs all
# dependencies using the environment's own pip binary directly.
#
# Tested on:
#   CUDA 13.0 (PyTorch 2.11.0+cu130) — UWaterloo cluster, May 2026
#   Python 3.12, conda ≥ 23.x
#
# Usage (on the GPU login / compute node):
#   chmod +x setup_env_gpu.sh
#   ./setup_env_gpu.sh
#
# To target a different CUDA version set TORCH_CUDA_URL before running:
#   CUDA 12.4 → export TORCH_CUDA_URL=https://download.pytorch.org/whl/cu124
#   CUDA 12.1 → export TORCH_CUDA_URL=https://download.pytorch.org/whl/cu121
#
# After it finishes:
#   conda activate torchdisorder
#   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   sbatch scripts/slurm_lips_67_noLi.sh
# =============================================================================

set -e

# Load NVIDIA HPC SDK — provides CUDA toolkit and compiler
module load nvhpc26.3

ENV_NAME="torchdisorder"
PYTHON_VER="3.12"

# ── CUDA index URL (override via env var if needed) ───────────────────────
TORCH_CUDA_URL="${TORCH_CUDA_URL:-https://download.pytorch.org/whl/cu130}"

echo ""
echo "============================================================"
echo "  TorchDisorder GPU Environment Setup"
echo "  Platform : $(uname -s) $(uname -m)"
echo "  Python   : $PYTHON_VER"
echo "  Env name : $ENV_NAME"
echo "  CUDA URL : $TORCH_CUDA_URL"
echo "============================================================"
echo ""

# ── 1. Remove existing env if present ────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "[1/6] Removing existing '$ENV_NAME' environment..."
    conda env remove -n "$ENV_NAME" -y
else
    echo "[1/6] No existing '$ENV_NAME' environment — skipping removal."
fi

# ── 2. Create fresh environment ───────────────────────────────────────────────
echo ""
echo "[2/6] Creating conda environment '$ENV_NAME' with Python $PYTHON_VER..."
conda create -n "$ENV_NAME" python="$PYTHON_VER" pip -y

# Ask conda where it actually installed the environment (may differ from conda base)
ENV_PATH="$(conda env list | grep "^${ENV_NAME}[[:space:]]" | awk '{print $NF}')"
PYTHON="$ENV_PATH/bin/python"

echo "  env    : $ENV_PATH"
echo "  python : $PYTHON"
$PYTHON -m pip install --upgrade pip wheel setuptools

# ── 3. PyTorch with CUDA ──────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing PyTorch (CUDA 13.0)..."
$PYTHON -m pip install \
    "torch" \
    "torchvision" \
    "torchaudio" \
    --index-url "$TORCH_CUDA_URL"

# ── 4. Core simulation stack ──────────────────────────────────────────────────
echo ""
echo "[4/6] Installing core simulation stack..."

$PYTHON -m pip install \
    "torch-sim-atomistic" \
    "cooper-optim"

# ── 5. All remaining dependencies ────────────────────────────────────────────
echo ""
echo "[5/6] Installing all TorchDisorder dependencies..."

$PYTHON -m pip install \
    "ase" \
    "pymatgen" \
    "mace-torch" \
    "spglib" \
    "matscipy" \
    "vesin" \
    "vesin-torch" \
    "omegaconf" \
    "hydra-core" \
    "pytorch-lightning" \
    "torchmetrics" \
    "lightning-utilities" \
    "wandb" \
    "pandas" \
    "numpy" \
    "scipy" \
    "matplotlib" \
    "scikit-learn" \
    "plotly" \
    "dash" \
    "nvidia-ml-py3" \
    "psutil" \
    "warp-lang" \
    "e3nn" \
    "tqdm" \
    "pyyaml" \
    "monty" \
    "tabulate" \
    "prettytable" \
    "lmdb" \
    "h5py" \
    "Pillow" \
    "pytest" \
    "pytest-cov" \
    "ruff" \
    "nvalchemi-toolkit-ops"

# ── 6. Install torchdisorder package itself (editable) ───────────────────────
echo ""
echo "[6/6] Installing torchdisorder package (editable mode)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$PYTHON -m pip install -e "$SCRIPT_DIR" --no-deps

# ── 6b. Patch torch_sim for PyTorch 2.7+ TorchScript compatibility ────────────
# torch.nonzero(x, as_tuple=True) is not in the TorchScript ATen schema for
# PyTorch >= 2.7. Replace with torch.where(x) which is equivalent and works.
echo ""
echo "[patch] Fixing torch_sim TorchScript compatibility (nonzero as_tuple)..."
$PYTHON - <<'PYEOF'
import re, os, sys
try:
    import torch_sim, inspect
    f = os.path.join(os.path.dirname(inspect.getfile(torch_sim)), "transforms.py")
    txt = open(f).read()
    txt2 = re.sub(r"torch\.nonzero\((\w+),\s*as_tuple=True\)", r"torch.where(\1)", txt)
    n = txt.count("as_tuple") - txt2.count("as_tuple")
    if n > 0:
        open(f, "w").write(txt2)
        print(f"  Patched {n} occurrence(s) in {f}")
    else:
        print("  No patch needed (already fixed or different version).")
except Exception as e:
    print(f"  WARNING: patch failed — {e}")
    sys.exit(0)
PYEOF

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  GPU Environment Setup Complete!"
echo "============================================================"
echo ""
echo "  Activate:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Quick GPU test:"
echo "    python -c \"import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\""
echo ""
echo "  Submit SLURM jobs (from project root):"
echo "    sbatch scripts/slurm_lips_67_noLi.sh"
echo "    sbatch scripts/slurm_lips_67_withLi.sh"
echo "    sbatch scripts/slurm_lips_70_noLi.sh"
echo "    sbatch scripts/slurm_lips_70_withLi.sh"
echo "    sbatch scripts/slurm_lips_75_noLi.sh"
echo "    sbatch scripts/slurm_lips_75_withLi.sh"
echo ""
echo "  NOTE: If your cluster has a different CUDA version:"
echo "    export TORCH_CUDA_URL=https://download.pytorch.org/whl/cu124"
echo "    ./setup_env_gpu.sh"
echo "============================================================"
