"""
Glass Structure Generator for Li-P-S System
Creates amorphous structures with specific P environment distributions
and generates order parameter constraints  simulations

Usage:
    python glass_generator.py --input Li7P3S11.cif --target 67Li2S-33P2S5 --supercell 5,8,5

# For 67Li2S-33P2S5 from Li7P3S11
python glass_generator.py \
  --input Li7P3S11.cif \
  --target 67Li2S-33P2S5 \
  --supercell 5,8,5 \
  --output glass_67Li2S

# For 70Li2S-30P2S5 from Li7P3S11
python glass_generator.py \
  --input Li7P3S11.cif \
  --target 70Li2S-30P2S5 \
  --supercell 5,8,5 \
  --output glass_70Li2S

# For 75Li2S-25P2S5 from β-Li3PS4
python glass_generator.py \
  --input Li3PS4_beta.cif \
  --target 75Li2S-25P2S5 \
  --supercell 5,6,9 \
  --output glass_75Li2S
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.local_env import CrystalNN
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import json


class PEnvironmentClassifier:
    """
    Classify P-S environments in Li-P-S structures.

    Environment types:
    - P4 (PS4^3-): P with 4 terminal S atoms
    - Pa/dimer (P2S7^4-): P with 3 terminal S + 1 bridging S (Sc)
    - P2/dumbbell (P2S6^4-): P with 3 terminal S + 1 P neighbor (P-P bond)
    """

    def __init__(self, structure: Structure, cutoff: float = 3.0):
        self.structure = structure
        self.cutoff = cutoff
        self.nn_finder = CrystalNN()

    def get_neighbors(self, site_index: int) -> Dict[str, List[int]]:
        """Get neighboring atoms categorized by element."""
        neighbors = {'P': [], 'S': [], 'Li': []}

        site = self.structure[site_index]
        for i, other_site in enumerate(self.structure):
            if i == site_index:
                continue

            distance = site.distance(other_site)
            if distance < self.cutoff:
                element = other_site.specie.symbol
                if element in neighbors:
                    neighbors[element].append(i)

        return neighbors

    def classify_p_site(self, p_index: int) -> Dict:
        """
        Classify a single P site.

        Returns:
            dict with keys: 'type', 'n_p', 'n_s', 'neighbors', 'label'
        """
        neighbors = self.get_neighbors(p_index)
        n_p = len(neighbors['P'])
        n_s = len(neighbors['S'])

        # Classify based on coordination
        if n_p == 0 and n_s == 4:
            env_type = 'P4'
            label = 'PS4^3-'
        elif n_p == 0 and n_s == 3:
            env_type = 'P3'
            label = 'PS3^-'
        elif n_p == 1 and n_s == 3:
            # Check if it's dumbbell (P2S6) or dimer (P2S7)
            # For P2S7: the P atoms share a bridging S
            # For P2S6: direct P-P bond, no shared S
            p_neighbor_idx = neighbors['P'][0]
            p_neighbor_s = self.get_neighbors(p_neighbor_idx)['S']

            # Check for shared S atoms (bridging)
            shared_s = set(neighbors['S']).intersection(p_neighbor_s)

            if len(shared_s) > 0:
                env_type = 'Pa'
                label = 'P2S7^4- (dimer)'
            else:
                env_type = 'P2'
                label = 'P2S6^4- (dumbbell)'
        else:
            env_type = 'P_unknown'
            label = f'Unknown (P{n_p}S{n_s})'

        return {
            'type': env_type,
            'label': label,
            'n_p': n_p,
            'n_s': n_s,
            'neighbors': neighbors,
            'coordination': n_p + n_s
        }

    def classify_all_p(self) -> Dict[int, Dict]:
        """Classify all P sites in the structure."""
        classifications = {}

        for i, site in enumerate(self.structure):
            if site.specie.symbol == 'P':
                classifications[i] = self.classify_p_site(i)

        return classifications

    def get_statistics(self, classifications: Dict) -> Dict:
        """Calculate statistics of P environments."""
        counts = defaultdict(int)

        for data in classifications.values():
            counts[data['type']] += 1

        # For dimers and dumbbells, divide by 2 (2 P atoms per unit)
        total_units = (counts['P4'] + counts['P3'] +
                       counts['Pa'] // 2 + counts['P2'] // 2)

        fractions = {}
        if total_units > 0:
            fractions['PS4^3-'] = 100.0 * counts['P4'] / total_units
            fractions['PS3^-'] = 100.0 * counts['P3'] / total_units
            fractions['P2S7^4- (dimer)'] = 100.0 * (counts['Pa'] // 2) / total_units
            fractions['P2S6^4- (dumbbell)'] = 100.0 * (counts['P2'] // 2) / total_units

        return {
            'counts': dict(counts),
            'fractions': fractions,
            'total_p': len(classifications),
            'total_units': total_units
        }


class GlassStructureGenerator:
    """Generate glass structures with specific P environment distributions."""

    # Order parameter targets for each P environment type
    ENVIRONMENT_ORDER_PARAMETERS = {
        'P4': {  # PS4^3- tetrahedral
            'order_parameters': {
                'cn': {'target': 4.0, 'tolerance': 0.5, 'weight': 1.5,
                       'description': 'P-S coordination number'},
                'tet': {'target': 0.85, 'min': 0.7, 'max': 1.0, 'weight': 2.0,
                        'description': 'Tetrahedral PS4^3-'},
                'q4': {'target': 0.6, 'min': 0.4, 'max': 0.8, 'weight': 0.5,
                       'description': 'Tetrahedral bond order'}
            },
            'element_filter': [15, 16],  # P and S only
            'cutoff': 3.5
        },
        'Pa': {  # P2S7^4- dimer with bridging S
            'order_parameters': {
                'cn': {'target': 4.0, 'tolerance': 0.5, 'weight': 1.0,
                       'description': 'P coordination (3S + 1P)'},
                'tet': {'target': 0.6, 'min': 0.4, 'max': 0.8, 'weight': 1.5,
                        'description': 'Distorted tetrahedral (bridging S)'},
            },
            'element_filter': [15, 16],
            'cutoff': 3.5,
            'has_p_neighbor': True,
            'expected_p_p_distance': {'min': 2.8, 'max': 3.5}  # Bridged P-P
        },
        'P2': {  # P2S6^4- dumbbell with P-P bond
            'order_parameters': {
                'cn': {'target': 4.0, 'tolerance': 0.5, 'weight': 1.0,
                       'description': 'P coordination (3S + 1P)'},
                'tet': {'target': 0.65, 'min': 0.4, 'max': 0.85, 'weight': 1.5,
                        'description': 'Tetrahedral with P-P bond'},
            },
            'element_filter': [15, 16],
            'cutoff': 3.5,
            'has_p_neighbor': True,
            'expected_p_p_distance': {'min': 2.0, 'max': 2.4}  # Direct P-P bond
        },
        'P3': {  # PS3^- pyramidal
            'order_parameters': {
                'cn': {'target': 3.0, 'tolerance': 0.5, 'weight': 1.5,
                       'description': 'P-S coordination'},
                'q2': {'target': 0.5, 'min': 0.3, 'max': 0.7, 'weight': 1.0,
                       'description': 'Pyramidal symmetry'}
            },
            'element_filter': [15, 16],
            'cutoff': 3.5
        }
    }

    TARGET_COMPOSITIONS = {
        '67Li2S-33P2S5': {
            'P2S6^4- (dumbbell)': 14.2,
            'P2S7^4- (dimer)': 78.5,
            'PS4^3-': 7.3,
            'S2-': 0.0
        },
        '70Li2S-30P2S5': {
            'P2S6^4- (dumbbell)': 12.9,
            'P2S7^4- (dimer)': 58.6,
            'PS4^3-': 28.5,
            'S2-': 0.0
        },
        '75Li2S-25P2S5': {
            'P2S6^4- (dumbbell)': 13.8,
            'P2S7^4- (dimer)': 12.6,
            'PS4^3-': 73.6,
            'S2-': 0.0
        }
    }

    def __init__(self, structure: Structure, target_composition: str):
        self.structure = structure.copy()
        self.target = self.TARGET_COMPOSITIONS[target_composition]
        self.target_composition = target_composition
        self.classifier = PEnvironmentClassifier(self.structure)

    def remove_li_and_reindex(self, classifications: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Remove all Li atoms from self.structure and remap indices in classifications
        (including neighbor index lists) to match the new Structure indexing.
        """
        # Build old->new index map for all sites
        old_to_new: Dict[int, int] = {}
        keep_old_indices: List[int] = []

        for i, site in enumerate(self.structure):
            if site.specie.symbol == "Li":
                old_to_new[i] = -1
            else:
                keep_old_indices.append(i)

        for new_i, old_i in enumerate(keep_old_indices):
            old_to_new[old_i] = new_i

        # Remove all Li in-place; indices will be compacted/reindexed by pymatgen.
        self.structure.remove_species(["Li+"])

        # Refresh classifier
        self.classifier = PEnvironmentClassifier(self.structure)

        # Remap classifications dict (keys and neighbor lists)
        new_classifications: Dict[int, Dict] = {}

        for old_p_idx, data in classifications.items():
            new_p_idx = old_to_new.get(old_p_idx, -1)
            if new_p_idx == -1:
                continue

            new_data = dict(data)
            new_neighbors = {k: [] for k in data["neighbors"].keys()}

            for el, neigh_list in data["neighbors"].items():
                for old_n in neigh_list:
                    new_n = old_to_new.get(old_n, -1)
                    if new_n != -1:
                        new_neighbors[el].append(new_n)

            new_data["neighbors"] = new_neighbors
            new_classifications[new_p_idx] = new_data

        return new_classifications

    def create_supercell(self, scaling_matrix: List[int]):
        """Create supercell."""
        self.structure.make_supercell(scaling_matrix)
        self.classifier = PEnvironmentClassifier(self.structure)
        print(f"Created supercell: {self.structure.num_sites} atoms")

    def identify_current_environments(self) -> Dict:
        """Identify current P environments in structure."""
        classifications = self.classifier.classify_all_p()
        stats = self.classifier.get_statistics(classifications)

        print("\n" + "=" * 60)
        print("CURRENT P ENVIRONMENT DISTRIBUTION")
        print("=" * 60)
        print(f"Total P atoms: {stats['total_p']}")
        print(f"Total anionic units: {stats['total_units']}")
        print("\nFractions:")
        for env, frac in stats['fractions'].items():
            print(f"  {env:30s}: {frac:6.2f}%")

        return classifications, stats

    def convert_p4_to_dumbbell(self, p4_indices: List[int], n_convert: int):
        """
        Convert PS4^3- units to P2S6^4- (dumbbell) units.

        Strategy: Find pairs of P4 sites, remove one S from each,
                 bring them close to form P-P bond.
        """
        converted = 0
        p4_available = list(p4_indices)
        np.random.shuffle(p4_available)

        print(f"\nConverting {n_convert} P4 pairs → P2 (dumbbell)...")

        while converted < n_convert and len(p4_available) >= 2:
            # Take two P4 sites
            idx1 = p4_available.pop()
            idx2 = p4_available.pop()

            # Get their S neighbors
            neighs1 = self.classifier.get_neighbors(idx1)['S']
            neighs2 = self.classifier.get_neighbors(idx2)['S']

            if len(neighs1) >= 4 and len(neighs2) >= 4:
                # Remove one S from each (choose furthest)
                site1 = self.structure[idx1]
                site2 = self.structure[idx2]

                # Find furthest S from each P
                dists1 = [(i, site1.distance(self.structure[i])) for i in neighs1]
                dists2 = [(i, site2.distance(self.structure[i])) for i in neighs2]

                s_remove1 = max(dists1, key=lambda x: x[1])[0]
                s_remove2 = max(dists2, key=lambda x: x[1])[0]

                # Move P atoms closer (simulate P-P bond formation ~2.2 Å)
                vec = site2.coords - site1.coords
                distance = np.linalg.norm(vec)

                if distance > 2.5:  # Only if not already close
                    # Move both P towards each other
                    move_dist = (distance - 2.2) / 2
                    direction = vec / distance

                    self.structure.translate_sites(
                        [idx1], move_dist * direction, frac_coords=False
                    )
                    self.structure.translate_sites(
                        [idx2], -move_dist * direction, frac_coords=False
                    )

                # Remove the S atoms
                indices_to_remove = sorted([s_remove1, s_remove2], reverse=True)
                for idx in indices_to_remove:
                    self.structure.remove_sites([idx])

                converted += 1

                # Update classifier with new structure
                self.classifier = PEnvironmentClassifier(self.structure)

        print(f"  Successfully converted {converted} pairs")
        return converted

    def convert_p4_to_dimer(self, p4_indices: List[int], n_convert: int):
        """
        Convert PS4^3- units to P2S7^4- (dimer) units.

        Strategy: Find pairs of P4 sites, merge them sharing one S
                 as bridging atom (Sc type).
        """
        converted = 0
        p4_available = list(p4_indices)
        np.random.shuffle(p4_available)

        print(f"\nConverting {n_convert} P4 pairs → Pa (dimer)...")

        while converted < n_convert and len(p4_available) >= 2:
            # Take two P4 sites
            idx1 = p4_available.pop()
            idx2 = p4_available.pop()

            # Get their S neighbors
            neighs1 = self.classifier.get_neighbors(idx1)['S']
            neighs2 = self.classifier.get_neighbors(idx2)['S']

            if len(neighs1) >= 4 and len(neighs2) >= 4:
                site1 = self.structure[idx1]
                site2 = self.structure[idx2]

                # Pick one S from each to become bridging
                s1_idx = neighs1[0]
                s2_idx = neighs2[0]

                s1_site = self.structure[s1_idx]
                s2_site = self.structure[s2_idx]

                # Move S atoms to midpoint (bridging position)
                midpoint = (s1_site.coords + s2_site.coords) / 2

                self.structure.translate_sites(
                    [s1_idx], midpoint - s1_site.coords, frac_coords=False
                )

                # Remove the second S (now we have shared bridging S)
                self.structure.remove_sites([s2_idx])

                # Move P atoms closer to bridging S
                # Each P should be ~2.1 Å from bridging S
                for p_idx in [idx1, idx2]:
                    if p_idx < len(self.structure):  # Check still valid after removal
                        p_site = self.structure[p_idx]
                        vec_to_s = midpoint - p_site.coords
                        dist = np.linalg.norm(vec_to_s)
                        if dist > 2.3:
                            move_vec = vec_to_s * (1 - 2.1 / dist)
                            self.structure.translate_sites(
                                [p_idx], move_vec, frac_coords=False
                            )

                converted += 1

                # Update classifier
                self.classifier = PEnvironmentClassifier(self.structure)

        print(f"  Successfully converted {converted} pairs")
        return converted

    def generate_glass_structure(self):
        """Generate glass structure with target environment distribution."""
        print("\n" + "=" * 60)
        print("GENERATING GLASS STRUCTURE")
        print("=" * 60)

        # Get current state
        classifications, current_stats = self.identify_current_environments()

        # Calculate what needs to change
        print("\n" + "=" * 60)
        print("TARGET DISTRIBUTION")
        print("=" * 60)
        for env, frac in self.target.items():
            current = current_stats['fractions'].get(env, 0.0)
            print(f"  {env:30s}: {current:6.2f}% → {frac:6.2f}%")

        # Identify P sites by type
        p4_sites = [idx for idx, data in classifications.items()
                    if data['type'] == 'P4']
        pa_sites = [idx for idx, data in classifications.items()
                    if data['type'] == 'Pa']
        p2_sites = [idx for idx, data in classifications.items()
                    if data['type'] == 'P2']

        total_units = current_stats['total_units']

        # Calculate number of conversions needed
        target_p2_units = int(total_units * self.target['P2S6^4- (dumbbell)'] / 100)
        target_pa_units = int(total_units * self.target['P2S7^4- (dimer)'] / 100)

        current_p2_units = len(p2_sites) // 2
        current_pa_units = len(pa_sites) // 2

        n_p2_convert = max(0, target_p2_units - current_p2_units)
        n_pa_convert = max(0, target_pa_units - current_pa_units)

        print(f"\nConversions needed:")
        print(f"  P4 → P2 (dumbbell): {n_p2_convert} pairs")
        print(f"  P4 → Pa (dimer): {n_pa_convert} pairs")

        # Perform conversions
        if n_p2_convert > 0:
            self.convert_p4_to_dumbbell(p4_sites, n_p2_convert)

        if n_pa_convert > 0:
            self.convert_p4_to_dimer(p4_sites, n_pa_convert)

        # Verify final state
        print("\n" + "=" * 60)
        print("FINAL P ENVIRONMENT DISTRIBUTION")
        print("=" * 60)
        final_class, final_stats = self.identify_current_environments()

        return final_class, final_stats

    def add_disorder(self, displacement: float = 0.3):
        """Add random atomic displacements to create amorphous structure."""
        print(f"\nAdding disorder (max displacement: {displacement} Å)...")

        for i in range(len(self.structure)):
            # Random displacement
            disp = np.random.randn(3) * displacement
            self.structure.translate_sites([i], disp, frac_coords=False)

        print("  Disorder added")

    def generate_order_parameter_constraints(self, classifications: Dict, stats: Dict) -> Dict:
        """
        Generate per-atom order parameter constraints for augmented Lagrangian

        Returns:
            Dictionary with constraint specifications for each P atom
        """
        constraints = {
            'cutoff': 3.5,  # Distance cutoff for neighbor finding (Å)
            'element_filter': [15, 16],  # P=15, S=16 atomic numbers
            'atom_constraints': {}
        }

        for p_idx, data in classifications.items():
            env_type = data['type']

            # Skip unknown environments
            if env_type not in self.ENVIRONMENT_ORDER_PARAMETERS:
                continue

            # Get environment-specific parameters
            env_params = self.ENVIRONMENT_ORDER_PARAMETERS[env_type]

            atom_constraint = {
                'atom_index': p_idx,
                'element': 'P',
                'environment_type': env_type,
                'environment_label': data['label'],
                'target_coordination': data['coordination'],
                'order_parameters': {}
            }

            # Copy order parameter constraints for this environment
            for op_name, op_params in env_params['order_parameters'].items():
                atom_constraint['order_parameters'][op_name] = op_params.copy()

            # Add connectivity information
            if env_params.get('has_p_neighbor', False):
                atom_constraint['has_p_neighbor'] = True
                if 'expected_p_p_distance' in env_params:
                    atom_constraint['expected_p_p_distance'] = env_params['expected_p_p_distance']
                atom_constraint['p_neighbor_indices'] = data['neighbors']['P']

            # Add neighbor information
            atom_constraint['s_neighbor_indices'] = data['neighbors']['S']
            atom_constraint['n_p'] = data['n_p']
            atom_constraint['n_s'] = data['n_s']

            constraints['atom_constraints'][str(p_idx)] = atom_constraint

        # Add global constraints
        constraints['global_constraints'] = {
            'p_environment_distribution': {
                'target_fractions': self.target,
                'current_fractions': stats['fractions'],
                'description': 'Maintain P environment distribution'
            },
            'total_p_atoms': stats['total_p'],
            'total_anionic_units': stats['total_units'],
            'target_composition': self.target_composition
        }

        # Add metadata
        constraints['metadata'] = {
            'structure_type': 'li_p_s_glass',
            'target_composition': self.target_composition,
            'total_atoms': self.structure.num_sites,
            'composition': str(self.structure.composition),
            'cutoff_description': 'Distance cutoff for P-S neighbor finding',
            'order_parameter_types': list(set(
                op for env in self.ENVIRONMENT_ORDER_PARAMETERS.values()
                for op in env['order_parameters'].keys()
            )),
            'notes': 'Constraints for augmented Lagrangian   with TorchSimOrderParameters',
            'environment_types': {
                'P4': 'Tetrahedral PS4^3-',
                'Pa': 'P2S7^4- dimer with bridging S',
                'P2': 'P2S6^4- dumbbell with P-P bond',
                'P3': 'Pyramidal PS3^-'
            }
        }

        return constraints

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def write_output(self, output_prefix: str, classifications: Dict, stats: Dict):
        """Write output CIF, summary, and constraint files."""
        # Write CIF
        cif_file = f"{output_prefix}.cif"
        writer = CifWriter(self.structure)
        writer.write_file(cif_file)
        print(f"\nWrote structure to: {cif_file}")

        # Write order parameter constraints for augmented Lagrangian
        constraints_file = f"{output_prefix}_constraints.json"
        constraints = self.generate_order_parameter_constraints(classifications, stats)

        with open(constraints_file, 'w') as f:
            json.dump(constraints, f, indent=2)

        print(f"Wrote   constraints to: {constraints_file}")
        print(f"  - {len(constraints['atom_constraints'])} P atoms with constraints")
        print(f"  - Order parameters: {', '.join(constraints['metadata']['order_parameter_types'])}")

        # Write detailed P environment summary
        summary_file = f"{output_prefix}_P_environments.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("P ENVIRONMENT SUMMARY FOR  \n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total atoms: {self.structure.num_sites}\n")
            f.write(f"Total P atoms: {stats['total_p']}\n")
            f.write(f"Total anionic units: {stats['total_units']}\n\n")

            f.write("Environment Distribution:\n")
            f.write("-" * 70 + "\n")
            for env, frac in stats['fractions'].items():
                count = stats['counts'].get(env.split()[0].replace('S', '').replace('^', ''), 0)
                f.write(f"  {env:35s}: {frac:6.2f}% ({count} P atoms)\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("ORDER PARAMETER CONSTRAINTS BY ENVIRONMENT\n")
            f.write("=" * 70 + "\n\n")

            for env_type, env_params in self.ENVIRONMENT_ORDER_PARAMETERS.items():
                count = stats['counts'].get(env_type, 0)
                if count > 0:
                    f.write(f"\n{env_type} environments ({count} atoms):\n")
                    f.write("-" * 70 + "\n")
                    for op_name, op_params in env_params['order_parameters'].items():
                        f.write(f"  {op_name}: target={op_params['target']:.2f}, ")
                        if 'min' in op_params:
                            f.write(f"range=[{op_params['min']:.2f}, {op_params['max']:.2f}], ")
                        f.write(f"weight={op_params['weight']:.1f}\n")
                        if 'description' in op_params:
                            f.write(f"      ({op_params['description']})\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("DETAILED P SITE INFORMATION\n")
            f.write("=" * 70 + "\n\n")

            # Group by type
            by_type = defaultdict(list)
            for idx, data in classifications.items():
                by_type[data['type']].append((idx, data))

            for ptype in ['P4', 'P2', 'Pa', 'P3']:
                if ptype in by_type:
                    sites = by_type[ptype]
                    f.write(f"\n{ptype} sites ({len(sites)} atoms):\n")
                    f.write("-" * 70 + "\n")

                    for idx, data in sites:
                        coords = self.structure[idx].coords
                        f.write(f"  Index {idx:4d}: {data['label']:25s} ")
                        f.write(f"CN={data['coordination']} ")
                        f.write(f"({data['n_p']}P + {data['n_s']}S) ")
                        f.write(f"xyz=({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})\n")

                        # List neighbors
                        neighs = data['neighbors']
                        if neighs['P']:
                            f.write(f"      P neighbors: {neighs['P']}\n")
                        if neighs['S']:
                            f.write(f"      S neighbors: {neighs['S'][:8]}")
                            if len(neighs['S']) > 8:
                                f.write(f" ... ({len(neighs['S'])} total)")
                            f.write("\n")

        print(f"Wrote P environment summary to: {summary_file}")

        # Write JSON for programmatic access (convert numpy types)
        json_file = f"{output_prefix}_P_environments.json"
        json_data = {
            'statistics': self._convert_to_json_serializable(stats),
            'classifications': {
                str(k): {
                    'type': str(v['type']),
                    'label': str(v['label']),
                    'coordination': int(v['coordination']),
                    'n_p': int(v['n_p']),
                    'n_s': int(v['n_s']),
                    'coords': [float(x) for x in self.structure[k].coords]
                }
                for k, v in classifications.items()
            }
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Wrote JSON data to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate glass structures for   simulations'
    )
    parser.add_argument('--input', required=True, help='Input CIF file')
    parser.add_argument('--target', required=True,
                        choices=['67Li2S-33P2S5', '70Li2S-30P2S5', '75Li2S-25P2S5'],
                        help='Target glass composition')
    parser.add_argument('--supercell', default='5,8,5',
                        help='Supercell scaling (e.g., "5,8,5")')
    parser.add_argument('--output', default='glass_structure',
                        help='Output file prefix')
    parser.add_argument('--disorder', type=float, default=0.3,
                        help='Random displacement magnitude (Å)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load structure
    print(f"Loading structure from: {args.input}")
    structure = Structure.from_file(args.input)
    print(f"  Formula: {structure.composition.reduced_formula}")
    print(f"  Space group: {structure.get_space_group_info()}")
    print(f"  Number of sites: {structure.num_sites}")

    # Create generator
    generator = GlassStructureGenerator(structure, args.target)

    # Create supercell
    scaling = [int(x) for x in args.supercell.split(',')]
    generator.create_supercell(scaling)

    # Generate glass structure
    classifications, stats = generator.generate_glass_structure()

    # Add disorder
    generator.add_disorder(displacement=args.disorder)

    # ===== CHANGE: remove Li, then RECLASSIFY on Li-free structure (fresh indexing) =====
    generator.structure.remove_species(["Li+"])
    generator.classifier = PEnvironmentClassifier(generator.structure)
    classifications = generator.classifier.classify_all_p()
    stats = generator.classifier.get_statistics(classifications)
    # ================================================================================

    # Write output
    generator.write_output(args.output, classifications, stats)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  1. {args.output}.cif - Structure file ")
    print(f"  2. {args.output}_constraints.json - Order parameter constraints")
    print(f"  3. {args.output}_P_environments.txt - Human-readable summary")
    print(f"  4. {args.output}_P_environments.json - Machine-readable data")
    print("\nNext steps:")
    print("  1. Load structure and constraints in your augmented Lagrangian ")
    print("  2. Use TorchSimOrderParameters to compute order parameters")
    print("  3. Apply constraints during PDF/RDF matching")
    print("  4. Monitor P environment distributions during refinement")
    print("\nExample constraint usage:")
    print("  from order_params import TorchSimOrderParameters")
    print("  op_calc = TorchSimOrderParameters(cutoff=3.5, device='cuda')")
    print("  # Apply per-atom constraints during  optimization")


if __name__ == "__main__":
    main()
