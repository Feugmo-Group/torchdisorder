from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

# Read CIF file
structure = Structure.from_file('/home/advaitgore/PycharmProjects/torchdisorder/data-release/crystal-structures/GeO2_mp-223_conventional_standard.cif')

# Create 5x5x5 supercell
supercell = structure.make_supercell([5, 5, 5])

# Save to new CIF file
CifWriter(supercell).write_file('c-GeO2.cif')
