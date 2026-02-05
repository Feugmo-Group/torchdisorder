#!/bin/bash
# ==============================================================================
# TorchDisorder Environment Setup for HPC Clusters
# ==============================================================================
#
# This script sets up the Python environment for TorchDisorder on various
# HPC clusters (Compute Canada, etc.)
#
# Usage:
#   source setup_env.sh         # Setup and activate environment
#   source setup_env.sh --fresh # Delete and recreate environment
#
# ==============================================================================

set -e

VENV_PATH="$HOME/.venv/torchdisorder"
FRESH_INSTALL=false

if [[ "$1" == "--fresh" ]]; then
    FRESH_INSTALL=true
fi

echo "=============================================="
echo "TorchDisorder Environment Setup"
echo "=============================================="
echo ""

# Detect cluster
if [[ -n "$CC_CLUSTER" ]]; then
    echo "Detected Compute Canada cluster: $CC_CLUSTER"
    CLUSTER_TYPE="cc"
else
    echo "Using generic cluster setup"
    CLUSTER_TYPE="generic"
fi

# Load modules based on cluster
echo ""
echo "Loading modules..."

if [[ "$CLUSTER_TYPE" == "cc" ]]; then
    module purge
    module load StdEnv/2023
    module load python/3.11
    module load cuda/12.2
    module load cudnn/8.9
    module load scipy-stack/2023b
else
    # Generic module loading - adjust for your cluster
    module purge 2>/dev/null || true
    module load python/3.11 2>/dev/null || module load anaconda3 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi

echo "Modules loaded successfully"
echo ""

# Create or recreate virtual environment
if [[ "$FRESH_INSTALL" == true ]] && [[ -d "$VENV_PATH" ]]; then
    echo "Removing existing environment..."
    rm -rf "$VENV_PATH"
fi

if [[ ! -d "$VENV_PATH" ]]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python -m venv "$VENV_PATH"
    
    # Activate
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support
    echo ""
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install core dependencies
    echo ""
    echo "Installing TorchDisorder dependencies..."
    pip install \
        numpy \
        scipy \
        pandas \
        matplotlib \
        plotly \
        ase \
        pymatgen \
        hydra-core \
        omegaconf \
        wandb \
        vesin
    
    # Install torch-sim (may need special handling)
    echo ""
    echo "Installing torch-sim..."
    pip install torch-sim || echo "Warning: torch-sim installation may require manual setup"
    
    # Install cooper
    echo ""
    echo "Installing Cooper..."
    pip install cooper-optim || pip install git+https://github.com/cooper-org/cooper.git
    
    # Install MACE
    echo ""
    echo "Installing MACE..."
    pip install mace-torch || pip install git+https://github.com/ACEsuit/mace.git
    
    # Install torchdisorder in development mode
    echo ""
    echo "Installing TorchDisorder (development mode)..."
    pip install -e .
    
    echo ""
    echo "Environment setup complete!"
else
    echo "Activating existing environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Set environment variables
export PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Verify installation
echo ""
echo "=============================================="
echo "Environment Verification"
echo "=============================================="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    import torchdisorder
    print(f'TorchDisorder version: {torchdisorder.__version__}')
except ImportError as e:
    print(f'TorchDisorder import error: {e}')

try:
    import cooper
    print('Cooper: OK')
except ImportError:
    print('Cooper: NOT INSTALLED')

try:
    import torch_sim
    print('torch-sim: OK')
except ImportError:
    print('torch-sim: NOT INSTALLED')

try:
    import mace
    print('MACE: OK')
except ImportError:
    print('MACE: NOT INSTALLED')
"

echo ""
echo "=============================================="
echo "Environment ready! Run training with:"
echo "  python scripts/train.py"
echo "Or submit a job with:"
echo "  sbatch submit.sh"
echo "=============================================="
