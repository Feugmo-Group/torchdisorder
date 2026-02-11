"""
Engine subpackage – Optimization and order parameter calculations.

v6: Environment-based constraints with adaptive penalties
v5: Legacy optimizer (backward compatible)
"""

# v6 Order Parameters
from torchdisorder.engine.order_params import (
    TorchSimOrderParameters,
    PyTorchOrderParameters,
    load_constraints_from_json,
    get_atom_indices_from_constraints,
    WARP_AVAILABLE,
)

# v6 Constrained Optimization
from torchdisorder.engine.constrained_optimizer import (
    EnvironmentConstrainedOptimizer,
    AdaptivePenalty,
    ConstantPenalty,
    EnvironmentConstraint,
    ConstraintState,
)

# v5 Backward Compatibility
from torchdisorder.engine.optimizer import (
    StructureFactorCMPWithConstraints,
    perform_melt_quench,
    perform_fire_relaxation,
)

# Callbacks
from torchdisorder.engine.callbacks import (
    EarlyStoppingCallback,
    CheckpointCallback,
    PlateauDetector,
)

__all__ = [
    # Order Parameters
    'TorchSimOrderParameters',
    'PyTorchOrderParameters',
    'load_constraints_from_json',
    'get_atom_indices_from_constraints',
    'WARP_AVAILABLE',
    
    # v6 Constrained Optimization
    'EnvironmentConstrainedOptimizer',
    'AdaptivePenalty',
    'ConstantPenalty',
    'EnvironmentConstraint',
    'ConstraintState',
    
    # v5 Backward Compatibility
    'StructureFactorCMPWithConstraints',
    'perform_melt_quench',
    'perform_fire_relaxation',
    
    # Callbacks
    'EarlyStoppingCallback',
    'CheckpointCallback',
    'PlateauDetector',
]
