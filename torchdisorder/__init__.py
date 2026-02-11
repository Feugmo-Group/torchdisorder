"""
TorchDisorder v6 – Differentiable Structure Optimization from Scattering Data
==============================================================================

This package provides tools for optimizing atomic structures to match
experimental scattering data (neutron or X-ray) while satisfying
structural constraints.

Key Improvements in v6:
    1. Unified Scattering Module: Single interface for S(Q), F(Q), g(r), T(r)
    2. Environment-Based Constraints: Group by local environment, not OP type
    3. Adaptive Penalties: Penalties grow for persistent violations

Main Components:
    - model.scattering: Unified differentiable scattering calculations
    - model.xrd: XRD/neutron diffraction model
    - model.loss: Loss functions for optimization
    - engine.constrained_optimizer: Environment-based constrained optimization (v6)
    - engine.optimizer: Legacy optimizer (v5 compatibility)
    - engine.order_params: Order parameter calculations

Usage (v6 style):
    >>> from torchdisorder.model import XRDModel, CooperLoss
    >>> from torchdisorder.engine import EnvironmentConstrainedOptimizer
    >>> 
    >>> model = XRDModel(symbols, config, r_bins, q_bins)
    >>> loss_fn = CooperLoss(target_data, target_type='S_Q')
    >>> cmp = EnvironmentConstrainedOptimizer(...)

Usage (v5 style - backward compatible):
    >>> from torchdisorder import StructureFactorCMPWithConstraints
    >>> from torchdisorder.model.xrd import XRDModel
"""

__version__ = '0.6.0'
__author__ = 'Tetsassi Feugmo Research Group'

# =====================================================================
# v6 Core Imports
# =====================================================================
from torchdisorder.model.xrd import XRDModel
from torchdisorder.model.loss import CooperLoss, chi_squared, ChiSquaredObjective
from torchdisorder.model.scattering import (
    UnifiedSpectrumCalculator,
    SpectrumCalculator,
    ScatteringConfig,
)
from torchdisorder.engine.constrained_optimizer import (
    EnvironmentConstrainedOptimizer,
    AdaptivePenalty,
)
from torchdisorder.engine.order_params import TorchSimOrderParameters

# =====================================================================
# v5 Backward Compatibility Imports
# =====================================================================
from torchdisorder.engine.optimizer import (
    StructureFactorCMPWithConstraints,
    perform_melt_quench,
    perform_fire_relaxation,
    ConstantPenalty,
)
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.generator import generate_atoms_from_config

__all__ = [
    # Version
    '__version__',
    
    # v6 Models
    'XRDModel',
    'UnifiedSpectrumCalculator',
    'SpectrumCalculator',
    'ScatteringConfig',
    
    # Loss
    'CooperLoss',
    'chi_squared',
    'ChiSquaredObjective',
    
    # v6 Optimization
    'EnvironmentConstrainedOptimizer',
    'AdaptivePenalty',
    
    # v5 Backward Compatibility
    'StructureFactorCMPWithConstraints',
    'perform_melt_quench',
    'perform_fire_relaxation',
    'ConstantPenalty',
    'TargetRDFData',
    'generate_atoms_from_config',
    
    # Order Parameters
    'TorchSimOrderParameters',
]
