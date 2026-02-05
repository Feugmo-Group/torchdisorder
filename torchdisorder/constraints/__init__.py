"""
TorchDisorder Constraints Module

Constraint specification generators for different glass systems.

This module provides tools to generate JSON constraint files from crystalline
structures, specifying per-atom order parameter constraints for:
- SiO2 (silica glass) - tetrahedral Si environments
- GeO2 (germania glass) - tetrahedral Ge environments
- Li2S-P2S5 (lithium thiophosphate) - P4, Pa, P2 phosphorus environments
"""

from torchdisorder.constraints.sio2_generator import (
    SiEnvironmentClassifier,
    SiO2ConstraintWriter,
)

def generate_sio2_constraints(
    structure_file: str,
    output_prefix: str = "sio2",
    cutoff: float = 2.2
) -> dict:
    """
    Generate SiO2 constraint JSON from a structure file.
    
    Args:
        structure_file: Path to CIF/POSCAR structure file
        output_prefix: Prefix for output files
        cutoff: Si-O cutoff distance in Angstroms
        
    Returns:
        Dictionary of constraints
    """
    from pymatgen.core import Structure
    
    structure = Structure.from_file(structure_file)
    classifier = SiEnvironmentClassifier(structure, si_o_cutoff=cutoff)
    classifications = classifier.classify_all_si()
    stats = classifier.get_statistics(classifications)
    
    writer = SiO2ConstraintWriter(structure, classifier)
    writer.write_outputs(output_prefix, classifications, stats)
    
    return writer.generate_constraints(classifications, stats)

__all__ = [
    "SiEnvironmentClassifier",
    "SiO2ConstraintWriter",
    "generate_sio2_constraints",
]
