#!/usr/bin/env bash
# =============================================================================
# TorchDisorder – Conda Environment Setup  (macOS Apple Silicon / arm64)
# =============================================================================
#
# Creates a conda environment called "torchdisorder" with Python 3.12 and
# installs all packages needed to run training, the Faber-Ziman calculation,
# and the publication-quality plot scripts.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# After it finishes:
#   conda activate torchdisorder
#   python scripts/calc_fz_weights.py
#   python scripts/improve_plots.py
#   python scripts/train.py   # (requires GPU config adjustments for MPS)
# =============================================================================

set -e   # exit on first error

ENV_NAME="torchdisorder"
PYTHON_VER="3.12"

echo ""
echo "============================================================"
echo "  TorchDisorder Environment Setup"
echo "  Platform: $(uname -s) $(uname -m)"
echo "  Python:   $PYTHON_VER"
echo "  Env name: $ENV_NAME"
echo "============================================================"
echo ""

# ── 1. Remove existing env if present ─────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "[1/6] Removing existing '$ENV_NAME' environment..."
    conda env remove -n "$ENV_NAME" -y
else
    echo "[1/6] No existing '$ENV_NAME' environment found. Skipping removal."
fi

# ── 2. Create fresh environment ────────────────────────────────────────────────
echo ""
echo "[2/6] Creating conda environment '$ENV_NAME' with Python $PYTHON_VER..."
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

# ── 3. Activate ────────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Activating environment..."
# Use conda run instead of activate (works in non-interactive shells)
CONDA_RUN="conda run -n $ENV_NAME"

# Upgrade pip first
$CONDA_RUN pip install --upgrade pip wheel setuptools

# ── 4. PyTorch  (MPS backend for Apple Silicon – no CUDA needed) ───────────────
echo ""
echo "[4/6] Installing PyTorch 2.9.0 (MPS / CPU, Apple Silicon)..."
# On macOS arm64, the standard PyPI wheel includes MPS support.
# No extra --index-url needed.
$CONDA_RUN pip install \
    "torch>=2.9.0,<3.0.0" \
    "torchvision" \
    "torchaudio"

# ── 5. Core scientific stack ───────────────────────────────────────────────────
echo ""
echo "[5/6] Installing all TorchDisorder dependencies..."

# ── 5a. Pinned / version-sensitive packages ────────────────────────────────────
# NOTE: torch-sim-atomistic latest on PyPI is 0.3.0; the repo requirements.txt
#       lists 0.5.1 which is not yet published. Using 0.3.0 as the closest
#       available version. Update when 0.5.1 is released.
$CONDA_RUN pip install \
    "torch-sim-atomistic==0.3.0" \
    "cooper-optim>=1.0.1,<2.0.0"

# ── 5b. Atomistic simulation & structure ───────────────────────────────────────
$CONDA_RUN pip install \
    "ase>=3.27.0,<4.0.0" \
    "pymatgen>=2025.10.7,<2026.0.0" \
    "mace-torch"

# ── 5c. Configuration & training framework ─────────────────────────────────────
$CONDA_RUN pip install \
    "omegaconf>=2.3.0,<3.0.0" \
    "hydra-core>=1.3.2,<2.0.0" \
    "pytorch-lightning>=2.6.1,<3.0.0" \
    "torchmetrics>=1.8.2,<2.0.0"

# ── 5d. Experiment tracking ────────────────────────────────────────────────────
$CONDA_RUN pip install \
    "wandb>=0.24.2,<0.25.0"

# ── 5e. Data & analysis ────────────────────────────────────────────────────────
$CONDA_RUN pip install \
    "pandas>=2.0.0" \
    "numpy" \
    "scipy" \
    "matplotlib" \
    "scikit-learn"

# ── 5f. Visualisation ─────────────────────────────────────────────────────────
$CONDA_RUN pip install \
    "plotly>=5.0.0" \
    "dash>=2.0.0"

# ── 5g. Neighbour lists ───────────────────────────────────────────────────────
$CONDA_RUN pip install \
    "vesin>=0.4.2,<0.5.0" \
    "vesin-torch>=0.4.2,<0.5.0"

# ── 5h. NVIDIA Warp (Apple Silicon supported) ─────────────────────────────────
$CONDA_RUN pip install \
    "warp-lang>=1.11.1"

# ── 5i. GPU monitoring (installs fine on Mac; reports 0 GPUs) ─────────────────
$CONDA_RUN pip install \
    "nvidia-ml-py3"

# ── 5j. Notebook & widget support ─────────────────────────────────────────────
$CONDA_RUN pip install \
    "ipython>=9.0.0" \
    "ipywidgets>=8.0.0" \
    "anywidget>=0.9.0" \
    "jupyter" \
    "jupyterlab"

# ── 5k. Chemistry toolkit ─────────────────────────────────────────────────────
$CONDA_RUN pip install \
    "nvalchemi-toolkit-ops>=0.2.0,<0.3.0"

# ── 5l. Dev tools ─────────────────────────────────────────────────────────────
$CONDA_RUN pip install \
    "pytest>=8.0.0" \
    "pytest-cov>=6.0.0" \
    "ruff" \
    "psutil"

# ── 6. Install torchdisorder package itself (editable) ───────────────────────
echo ""
echo "[6/6] Installing torchdisorder package (editable mode)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$CONDA_RUN pip install -e "$SCRIPT_DIR" --no-deps

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Activate the environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Quick test:"
echo "    python -c \"import torch; print('PyTorch', torch.__version__); print('MPS available:', torch.backends.mps.is_available())\""
echo ""
echo "  Run Faber-Ziman calculation:"
echo "    cd $(dirname "${BASH_SOURCE[0]}")"
echo "    python scripts/calc_fz_weights.py"
echo ""
echo "  Improve paper plots:"
echo "    python scripts/improve_plots.py"
echo ""
echo "  NOTE: torch-sim-atomistic 0.5.1 is not yet on PyPI."
echo "        Installed 0.3.0 instead. Update when 0.5.1 is released."
echo "============================================================"
echo ""
