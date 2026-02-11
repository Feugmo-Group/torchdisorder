"""
Model subpackage – Differentiable models for scattering calculations.

v6: Unified scattering module with cleaner API
v5: Legacy rdf.py for backward compatibility
"""

# v6 Scattering (primary)
from torchdisorder.model.scattering import (
    UnifiedSpectrumCalculator,
    SpectrumCalculator,  # Alias to UnifiedSpectrumCalculator
    ScatteringConfig,
    ScatteringType,
    OutputType,
    get_distance_matrix,
    get_cell_volume,
)

# v5 Backward compatibility - import legacy rdf module components
from torchdisorder.model.rdf import (
    SpectrumCalculator as LegacySpectrumCalculator,
)

from torchdisorder.model.xrd import (
    XRDModel,
    XRDModelConfig,
    create_xrd_model,
)
from torchdisorder.model.loss import (
    CooperLoss,
    AugLagLoss,
    AugLagHyper,
    chi_squared,
    r_squared,
    rmse,
    ChiSquaredObjective,
)
from torchdisorder.model.generator import (
    generate_atoms_from_config,
)

__all__ = [
    # Scattering
    'UnifiedSpectrumCalculator',
    'SpectrumCalculator',
    'LegacySpectrumCalculator',
    'ScatteringConfig',
    'ScatteringType',
    'OutputType',
    'get_distance_matrix',
    'get_cell_volume',
    
    # XRD Model
    'XRDModel',
    'XRDModelConfig',
    'create_xrd_model',
    
    # Loss
    'CooperLoss',
    'AugLagLoss',
    'AugLagHyper',
    'chi_squared',
    'r_squared',
    'rmse',
    'ChiSquaredObjective',
    
    # Generator
    'generate_atoms_from_config',
]
