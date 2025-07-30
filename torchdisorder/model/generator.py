# model/generator.py

import numpy as np
from ase import Atoms
from ase.io.cif import read_cif
from pymatgen.core import Structure as PmgStructure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from typing import Literal, Tuple, Union

def generate_atoms_from_config(
    cfg,
    return_type: Literal["ase", "pymatgen"] = "ase"
) -> Union[Atoms, Structure]:
    """
    Generates a structure object from config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
    return_type : "ase" or "pymatgen"

    Returns
    -------
    ASE Atoms or pymatgen Structure depending on return_type
    """
    init_type: Literal["random_icp", "from_cif", "pymatgen"] = cfg.structure.init

    if init_type == "random_icp":
        atoms = generate_random_atoms(
            species=cfg.structure.species,
            stoichiometry=cfg.structure.stoichiometry,
            box_length=cfg.structure.box_length,
        )
        return atoms if return_type == "ase" else AseAtomsAdaptor.get_structure(atoms)

    elif init_type == "from_cif":
        atoms = generate_crystal_atoms(cfg.structure.cif_path, target_density=cfg.structure.target_density)
        return atoms if return_type == "ase" else AseAtomsAdaptor.get_structure(atoms)

    elif init_type == "pymatgen":
        struct, ase_atoms = generate_from_pymatgen(cfg.structure.cif_path, cfg.structure.target_density)
        return ase_atoms if return_type == "ase" else struct

    else:
        raise ValueError(f"Unknown structure.init = '{init_type}'")


def generate_random_atoms(species, stoichiometry, box_length) -> Atoms:
    assert len(species) == len(stoichiometry)
    total_particles = sum(stoichiometry)
    symbols = []
    for s, n in zip(species, stoichiometry):
        symbols.extend([s] * n)

    positions = np.random.rand(total_particles, 3) * box_length
    atoms = Atoms(symbols=symbols, positions=positions, cell=box_length * np.eye(3), pbc=True)
    return atoms


def generate_crystal_atoms(cif_path: str, target_density: float = 2.20) -> Atoms:
    atoms = read_cif(cif_path)

    if isinstance(atoms, list):
        raise IOError("read_cif produced list of atoms... expected single structure.")

    atoms.rattle(stdev=0.05)
    original_volume = atoms.get_volume()
    mass = atoms.get_masses().sum()  # amu
    actual_density = (mass / 6.022e23) / (original_volume * 1e-24)  # g/cm^3
    scale_factor = (actual_density / target_density) ** (1/3)
    atoms.set_cell(atoms.get_cell() * scale_factor, scale_atoms=True)

    return atoms


def generate_from_pymatgen(cif_path: str, target_density: float = 2.20) -> Tuple[Structure, Atoms]:
    parser = CifParser(cif_path)
    structure: Structure = parser.get_structures()[0]

    # Convert to ASE
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)
    ase_atoms.rattle(stdev=0.05)

    original_volume = ase_atoms.get_volume()
    mass = ase_atoms.get_masses().sum()
    actual_density = (mass / 6.022e23) / (original_volume * 1e-24)
    scale_factor = (actual_density / target_density) ** (1/3)
    ase_atoms.set_cell(ase_atoms.get_cell() * scale_factor, scale_atoms=True)

    # Return both pymatgen and ASE representations
    return structure, ase_atoms
