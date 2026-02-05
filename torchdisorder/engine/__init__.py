"""
TorchDisorder Engine Module

Optimization engine with Cooper constrained optimization and order parameters.
"""

from torchdisorder.engine.optimizer import (
    StructureFactorCMPWithConstraints,
    perform_melt_quench,
    perform_fire_relaxation,
    ConstantPenalty,
)
from torchdisorder.engine.order_params import TorchSimOrderParameters
from torchdisorder.engine.callbacks import PlateauDetector

__all__ = [
    "StructureFactorCMPWithConstraints",
    "perform_melt_quench",
    "perform_fire_relaxation",
    "ConstantPenalty",
    "TorchSimOrderParameters",
    "PlateauDetector",
]
