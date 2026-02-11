"""
Constraint Generators for TorchDisorder v6
==========================================

Modules for generating v6-compatible constraint files from crystal structures.
Each generator:
    - Creates supercells from CIF files (--supercell N or --replicate na nb nc)
    - Classifies atomic environments by coordination number
    - Outputs JSON constraint files for EnvironmentConstrainedOptimizer

Available generators:
    - sio2_generator: SiO2 glass (Si4/Si3/Si5/Si6 environments)
    - geo2_generator: GeO2 glass (Ge4/Ge3/Ge5/Ge6 environments)

Usage:
    # SiO2 with ~1000 atom supercell
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --output sio2_glass --supercell 1000
    
    # GeO2 with manual 3x3x3 replication
    python -m torchdisorder.constraints.geo2_generator --input c-GeO2.cif --output geo2_glass --replicate 3 3 3
    
    # Only tetrahedral environments
    python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --environments Si4 --output sio2_tet
"""

from torchdisorder.constraints.sio2_generator import (
    SiEnvironmentClassifier,
    SiO2ConstraintWriter,
    create_supercell,
)

from torchdisorder.constraints.geo2_generator import (
    GeEnvironmentClassifier,
    GeO2ConstraintWriter,
)

__all__ = [
    'SiEnvironmentClassifier',
    'SiO2ConstraintWriter',
    'GeEnvironmentClassifier',
    'GeO2ConstraintWriter',
    'create_supercell',
]
