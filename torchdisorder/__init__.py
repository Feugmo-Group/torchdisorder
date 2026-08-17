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
# Lazy exports (PEP 562)
# =====================================================================
# These used to be eager `from ... import ...` lines, which meant that importing
# ANY submodule pulled in the entire training stack -- omegaconf, cooper,
# pymatgen, pandas, plotly -- because Python runs this file first.
#
# That is not a tidiness point, it destroyed real work. The MLIP backends each
# need their own conda env (mace-torch pins e3nn==0.4.4 while MatterSim and
# SevenNet need >=0.6), and those envs carry only what the dynamics needs. A
# 1.5-hour LiPS-25 melt-quench on 2026-08-17 therefore finished its physics,
# reached `from torchdisorder.common.validation import validate_structure`, and
# died on ModuleNotFoundError: omegaconf -- discarding the run at the last step.
#
# Resolving names on demand keeps `from torchdisorder import XRDModel` working
# while letting the analysis helpers, which need only numpy and ASE, be imported
# in an env that has just numpy and ASE.
_LAZY_EXPORTS = {
    'XRDModel': 'torchdisorder.model.xrd',
    'CooperLoss': 'torchdisorder.model.loss',
    'chi_squared': 'torchdisorder.model.loss',
    'ChiSquaredObjective': 'torchdisorder.model.loss',
    'UnifiedSpectrumCalculator': 'torchdisorder.model.scattering',
    'SpectrumCalculator': 'torchdisorder.model.scattering',
    'ScatteringConfig': 'torchdisorder.model.scattering',
    'EnvironmentConstrainedOptimizer': 'torchdisorder.engine.constrained_optimizer',
    'AdaptivePenalty': 'torchdisorder.engine.constrained_optimizer',
    'TorchSimOrderParameters': 'torchdisorder.engine.order_params',
    'StructureFactorCMPWithConstraints': 'torchdisorder.engine.optimizer',
    'perform_melt_quench': 'torchdisorder.engine.optimizer',
    'perform_fire_relaxation': 'torchdisorder.engine.optimizer',
    'ConstantPenalty': 'torchdisorder.engine.optimizer',
    'TargetRDFData': 'torchdisorder.common.target_rdf',
    'generate_atoms_from_config': 'torchdisorder.model.generator',
}


def __getattr__(name):
    """Import the module owning `name` on first access, then cache the result."""
    try:
        module_path = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # subsequent lookups skip __getattr__ entirely
    return value


def __dir__():
    return sorted(__all__)

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
