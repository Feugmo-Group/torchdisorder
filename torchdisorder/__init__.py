"""
TorchDisorder: Differentiable Generation of Amorphous Atomic Structures

A PyTorch-based framework for generating physically realistic amorphous structures
by fitting experimental diffraction data while enforcing local coordination constraints.

Key Features:
- Gradient-based optimization using automatic differentiation
- Constrained optimization via Cooper library (augmented Lagrangian)
- Order parameter constraints (tetrahedral, octahedral, bond-orientational)
- Support for multiple target types: S(Q), F(Q), T(r), G(r), g(r)
- Integration with torch-sim, MACE, and Pymatgen

Example:
    >>> from torchdisorder import StructureFactorCMPWithConstraints
    >>> from torchdisorder.model import XRDModel, SpectrumCalculator
    >>> from torchdisorder.common import TargetRDFData
    >>> 
    >>> # Load target data and create model
    >>> rdf_data = TargetRDFData.from_dict(config, device='cuda')
    >>> xrd_model = XRDModel(spec_calc, rdf_data, dtype=torch.float32, device='cuda')
    >>> 
    >>> # Create constrained optimization problem
    >>> cmp = StructureFactorCMPWithConstraints(
    ...     model=xrd_model,
    ...     base_state=state,
    ...     target_vec=rdf_data,
    ...     constraints_file='constraints.json'
    ... )
"""

__version__ = "0.2.0"
__author__ = "Feugmo Group, University of Waterloo"

# Core exports
from torchdisorder.engine.optimizer import (
    StructureFactorCMPWithConstraints,
    perform_melt_quench,
    perform_fire_relaxation,
)
from torchdisorder.engine.order_params import TorchSimOrderParameters
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.xrd import XRDModel
from torchdisorder.model.rdf import SpectrumCalculator
from torchdisorder.model.loss import CooperLoss, ChiSquaredObjective
from torchdisorder.model.generator import generate_atoms_from_config

__all__ = [
    # Core classes
    "StructureFactorCMPWithConstraints",
    "TorchSimOrderParameters",
    "TargetRDFData",
    "XRDModel",
    "SpectrumCalculator",
    "CooperLoss",
    "ChiSquaredObjective",
    # Functions
    "generate_atoms_from_config",
    "perform_melt_quench",
    "perform_fire_relaxation",
]
