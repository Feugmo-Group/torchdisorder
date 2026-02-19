from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

# 1. Load the original unit cell structure from your CIF file
# Replace 'Li3PS4.gamma.cif' with the path to your file
original_structure = Structure.from_file('/home/advaitgore/PycharmProjects/torchdisorder/data-release/crystal-structures/Fe2O3.cif')

# 2. Create the supercell
# The make_supercell method accepts a tuple (a, b, c) for the repetition along each axis
supercell_structure = original_structure.make_supercell((5,5,2))

# 3. Write the new supercell structure to a new CIF file
writer = CifWriter(supercell_structure)
writer.write_file('Fe2O3.cif')

print("done")