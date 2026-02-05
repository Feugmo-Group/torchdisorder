"""
TorchDisorder Common Utilities

Core utilities for device management, project paths, and helper functions.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from ase import Atoms
from ase.io import write

# ============================================================================
# Project Root and Environment
# ============================================================================

MODELS_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Set environment variable so Hydra/OmegaConf can access it
os.environ["PROJECT_ROOT"] = str(MODELS_PROJECT_ROOT)

# Register eval resolver for OmegaConf
def _try_eval(s):
    """Custom resolver for OmegaConf that allows eval in config files."""
    try:
        return eval(s)
    except Exception as e:
        print(f"Calling eval on string {s} raised exception {e}")
        raise

OmegaConf.register_new_resolver("eval", _try_eval, replace=True)

# ============================================================================
# Device Utilities
# ============================================================================

@lru_cache
def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache
def get_pyg_device() -> torch.device:
    """Get device for PyG operations (MPS not supported, falls back to CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# Trajectory Writing
# ============================================================================

def write_trajectories(
    atoms: Atoms, 
    traj_path: Optional[str] = None, 
    xdatcar_path: Optional[str] = None
):
    """
    Write atomic structure to trajectory files.
    
    Args:
        atoms: ASE Atoms object
        traj_path: Path for ASE trajectory file (.traj)
        xdatcar_path: Path for VASP XDATCAR format
    """
    if traj_path:
        write(traj_path, atoms, format="traj", append=True)
    if xdatcar_path:
        write(xdatcar_path, atoms, format="vasp-xdatcar", append=True)


# ============================================================================
# Atomic Number Constants
# ============================================================================

SELECTED_ATOMIC_NUMBERS = [
    1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
]

MAX_ATOMIC_NUM = 100
