"""
Structure Descriptors
=====================

Computes structural descriptors for an ASE Atoms object:
  - Radial distribution functions (RDF / g(r))
  - Bond-length distributions
  - Bond-angle distributions
  - Per-atom coordination numbers
  - Per-atom order parameters via TorchSimOrderParameters (q4, q6, tet, F_IS)
  - Full Steinhardt bond-orientational order parameters (q_l, w_l)
  - Ring statistics via networkx

All heavy geometry is done in plain NumPy / ASE so that this module can be
imported without a GPU.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import networkx as nx

import ase
import ase.data

import torch
import torch_sim as ts

from torchdisorder.engine.order_params import TorchSimOrderParameters

__all__ = ["StructureDescriptors"]


# ---------------------------------------------------------------------------
# Helper: build a SimState from an ASE Atoms object
# ---------------------------------------------------------------------------

def _atoms_to_simstate(atoms: ase.Atoms, device: str = "cpu") -> ts.SimState:
    """Convert an ASE Atoms object to a torch_sim SimState."""
    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
    cell = torch.tensor(
        atoms.cell.array, dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, 3, 3)
    atomic_numbers = torch.tensor(
        atoms.numbers, dtype=torch.int32, device=device
    )
    masses = torch.tensor(
        [ase.data.atomic_masses[z] for z in atoms.numbers],
        dtype=torch.float32,
        device=device,
    )
    pbc = torch.ones(1, 3, dtype=torch.bool, device=device)
    system_idx = torch.zeros(len(atoms), dtype=torch.long, device=device)

    return ts.SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=pbc,
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StructureDescriptors:
    """
    Structural descriptors for an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
    cutoff : float
        Default cutoff used by order-parameter computations (Å).
    device : str
        Torch device for order-parameter calculations.
    """

    def __init__(
        self,
        atoms: ase.Atoms,
        cutoff: float = 5.0,
        device: str = "cpu",
    ) -> None:
        self.atoms = atoms
        self.cutoff = cutoff
        self.device = device
        self._nl_cache: dict[float, tuple] = {}  # cutoff → (i, j, D, dist)

    # ------------------------------------------------------------------
    # Accelerated neighbor list via vesin (compiled C extension, ~50× faster
    # than ase.neighborlist.NeighborList for large systems).
    # Results are cached per cutoff so methods sharing a cutoff pay only once.
    # ------------------------------------------------------------------

    def _neighbors_vesin(
        self, cutoff: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (i, j, D, dist) for all pairs within cutoff using vesin.

        D[k] is the Cartesian vector from atom i[k] to atom j[k] (Å),
        including PBC image shifts.  dist[k] = |D[k]|.
        Results are cached per cutoff value.
        """
        if cutoff in self._nl_cache:
            return self._nl_cache[cutoff]

        from vesin import NeighborList as VesinNL
        pos = self.atoms.positions.astype(np.float64)
        cell = np.array(self.atoms.cell, dtype=np.float64)
        pbc = list(self.atoms.pbc) if hasattr(self.atoms.pbc, '__iter__') else [True, True, True]

        nl = VesinNL(cutoff=float(cutoff), full_list=True)
        i_arr, j_arr, _S, D_arr = nl.compute(
            points=pos, box=cell, periodic=pbc, quantities="ijSD"
        )
        dist_arr = np.linalg.norm(D_arr, axis=1)
        result = (i_arr, j_arr, D_arr, dist_arr)
        self._nl_cache[cutoff] = result
        return result

    # ------------------------------------------------------------------
    # RDF
    # ------------------------------------------------------------------

    def rdf(
        self,
        species_pair: tuple[str, str],
        r_max: float = 8.0,
        n_bins: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Partial radial distribution function g(r) for a given species pair.

        Parameters
        ----------
        species_pair : (elem_A, elem_B)
        r_max : float
            Maximum r to include (Å).
        n_bins : int
            Number of histogram bins.

        Returns
        -------
        r_centers : np.ndarray, shape (n_bins,)
        g_r       : np.ndarray, shape (n_bins,)
        """
        elem_a, elem_b = species_pair
        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())

        idx_a = np.where(symbols == elem_a)[0]
        idx_b = np.where(symbols == elem_b)[0]

        if len(idx_a) == 0 or len(idx_b) == 0:
            r_edges = np.linspace(0, r_max, n_bins + 1)
            return 0.5 * (r_edges[:-1] + r_edges[1:]), np.zeros(n_bins)

        i_arr, j_arr, _D, dist_arr = self._neighbors_vesin(r_max)

        # Keep only A→B pairs
        mask = (
            np.isin(i_arr, idx_a) &
            np.isin(j_arr, idx_b) &
            (dist_arr > 0)
        )
        dists = dist_arr[mask]

        bin_edges = np.linspace(0, r_max, n_bins + 1)
        counts, _ = np.histogram(dists, bins=bin_edges)

        r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dr = r_max / n_bins
        volume = atoms.get_volume()
        n_a = len(idx_a)
        n_b = len(idx_b)
        rho_b = n_b / volume

        shell_vols = 4.0 * np.pi * r_centers**2 * dr
        ideal = n_a * rho_b * shell_vols
        g_r = np.where(ideal > 0, counts / ideal, 0.0)
        return r_centers, g_r

    # ------------------------------------------------------------------
    # Bond length distribution
    # ------------------------------------------------------------------

    def bond_length_distribution(
        self,
        pair: tuple[str, str],
        cutoff: float,
    ) -> np.ndarray:
        """
        All bond lengths (Å) between two element types within cutoff.

        Returns
        -------
        lengths : np.ndarray
        """
        elem_a, elem_b = pair
        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())

        idx_a = np.where(symbols == elem_a)[0]
        idx_b = np.where(symbols == elem_b)[0]

        if len(idx_a) == 0 or len(idx_b) == 0:
            return np.array([])

        i_arr, j_arr, _D, dist_arr = self._neighbors_vesin(cutoff)
        mask = (
            np.isin(i_arr, idx_a) &
            np.isin(j_arr, idx_b) &
            (dist_arr > 0)
        )
        return dist_arr[mask]

    # ------------------------------------------------------------------
    # Bond angle distribution
    # ------------------------------------------------------------------

    def bond_angle_distribution(
        self,
        central: str,
        neighbor: str,
        cutoff: float,
    ) -> np.ndarray:
        """
        All neighbor-central-neighbor angles (degrees) for ``central`` atoms.

        Returns
        -------
        angles_deg : np.ndarray
        """
        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())
        idx_c = np.where(symbols == central)[0]

        idx_n = np.where(symbols == neighbor)[0]
        i_arr, j_arr, D_arr, dist_arr = self._neighbors_vesin(cutoff)

        # unit vectors from central to neighbor
        safe_dist = np.where(dist_arr > 0, dist_arr, 1.0)
        uvecs = D_arr / safe_dist[:, None]   # (N_pairs, 3)

        angles = []
        for i in idx_c:
            # all neighbor-type bonds from this central atom
            mask = (i_arr == i) & np.isin(j_arr, idx_n) & (dist_arr > 0)
            vecs = uvecs[mask]   # (n_b, 3)
            n_b = len(vecs)
            if n_b < 2:
                continue
            # vectorised pairwise dot products
            dots = vecs @ vecs.T   # (n_b, n_b)
            triu_i, triu_j = np.triu_indices(n_b, k=1)
            cos_vals = np.clip(dots[triu_i, triu_j], -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_vals)))

        return np.concatenate(angles) if angles else np.array([])

    # ------------------------------------------------------------------
    # Coordination numbers
    # ------------------------------------------------------------------

    def coordination_numbers(
        self,
        central: str,
        neighbor: str,
        cutoff: float,
    ) -> np.ndarray:
        """
        Integer coordination numbers for each central-element atom.

        Returns
        -------
        cn : np.ndarray[int]
        """
        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())
        idx_c = np.where(symbols == central)[0]

        if len(idx_c) == 0:
            return np.array([], dtype=int)

        idx_n = np.where(symbols == neighbor)[0]
        i_arr, j_arr, _D, dist_arr = self._neighbors_vesin(cutoff)

        mask = np.isin(i_arr, idx_c) & np.isin(j_arr, idx_n) & (dist_arr > 0)
        i_filt = i_arr[mask]

        # count per central atom
        cn_counts = np.bincount(i_filt, minlength=len(atoms))
        return cn_counts[idx_c].astype(int)

    # ------------------------------------------------------------------
    # Per-atom order parameters
    # ------------------------------------------------------------------

    def order_params_per_atom(
        self,
        central_z: int,
        neighbor_z: Optional[int],
        cutoff: float,
        compute: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Per-atom order parameters via TorchSimOrderParameters.

        Parameters
        ----------
        central_z : int
            Atomic number of the central species.
        neighbor_z : int or None
            Atomic number of the neighbor species. None = all species.
        cutoff : float
            Neighbor cutoff (Å).
        compute : list of str
            Subset of ['q4', 'q6', 'tet', 'fis'] to compute.

        Returns
        -------
        dict mapping OP name to np.ndarray of shape (n_central_atoms,)
        """
        if compute is None:
            compute = ["q4", "q6", "tet", "fis"]

        atoms = self.atoms
        state = _atoms_to_simstate(atoms, device=self.device)

        # Indices of central atoms
        central_mask = (atoms.numbers == central_z)
        central_indices = np.where(central_mask)[0]
        if len(central_indices) == 0:
            return {op: np.array([]) for op in compute}

        atom_indices = torch.tensor(
            central_indices, dtype=torch.long, device=self.device
        )

        element_filter = [neighbor_z] if neighbor_z is not None else None

        calc = TorchSimOrderParameters(
            cutoff=cutoff,
            device=self.device,
            max_neighbors=64,
            backend="pytorch",
            fis_mode="variable_R",
        )

        with torch.no_grad():
            results = calc(
                state,
                atom_indices,
                order_params=compute,
                element_filter=element_filter,
            )

        return {k: v.cpu().numpy() for k, v in results.items()}

    # ------------------------------------------------------------------
    # Full Steinhardt bond-orientational order parameters
    # ------------------------------------------------------------------

    def steinhardt_ql(
        self,
        l: int,
        cutoff: float,
        neighbor_elements: list[str] | None = None,
    ) -> np.ndarray:
        """Full Steinhardt q_l per atom, summing all m from -l to +l.

        q_l = sqrt(4pi/(2l+1) * sum_m |q_lm|^2)
        q_lm = (1/N_b) * sum_j Y_lm(theta_ij, phi_ij)

        For l=4 or l=6, tries the warp-accelerated kernel first (when warp
        is available and running on a CUDA device), then falls back to the
        vectorised NumPy implementation in steinhardt_warp.py, and finally
        falls back to the original scipy loop.

        Parameters
        ----------
        l : int
            Degree of the spherical harmonic (e.g. 4 or 6).
        cutoff : float
            Neighbor cutoff distance (Angstrom).
        neighbor_elements : list of str or None
            If given, only neighbors whose element is in this list are used.

        Returns
        -------
        q_l : np.ndarray, shape (N_atoms,)
            q_l value per atom (over all atoms in the structure).
        """
        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())
        n_atoms = len(atoms)

        i_arr, j_arr, D_arr, dist_arr = self._neighbors_vesin(cutoff)

        # element filter
        if neighbor_elements is not None:
            ne_set = set(neighbor_elements)
            elem_mask = np.array([symbols[j] in ne_set for j in j_arr])
            i_arr, j_arr, D_arr, dist_arr = (
                i_arr[elem_mask], j_arr[elem_mask],
                D_arr[elem_mask], dist_arr[elem_mask],
            )

        valid = dist_arr > 0
        i_arr, j_arr, D_arr, dist_arr = (
            i_arr[valid], j_arr[valid], D_arr[valid], dist_arr[valid]
        )

        # --- fast path: warp kernel or vectorised NumPy (l=4 or l=6 only) ---
        if l in (4, 6):
            try:
                from torchdisorder.engine.steinhardt_warp import (
                    WARP_AVAILABLE,
                    _build_padded_arrays,
                    steinhardt_ql_pytorch,
                    steinhardt_ql_warp,
                )

                # Build padded arrays
                max_neighbors = 64
                vectors_padded, mask_padded = _build_padded_arrays(
                    i_arr, j_arr, D_arr, dist_arr, n_atoms, max_neighbors
                )

                # Choose warp or NumPy fallback
                if WARP_AVAILABLE and "cuda" in self.device.lower():
                    return steinhardt_ql_warp(
                        l, vectors_padded, mask_padded, device=self.device
                    ).astype(float)
                else:
                    return steinhardt_ql_pytorch(
                        l, vectors_padded, mask_padded
                    ).astype(float)
            except Exception as exc:
                warnings.warn(
                    f"steinhardt_warp fast path failed ({exc!r}), "
                    "falling back to scipy loop"
                )

        # --- scipy fallback (any l, or if fast path raised) ---
        from scipy.special import sph_harm_y

        uvecs = D_arr / dist_arr[:, None]

        q_l_vals = np.zeros(n_atoms, dtype=float)
        prefactor = 4.0 * np.pi / (2 * l + 1)

        for i in range(n_atoms):
            mask_i = (i_arr == i)
            vecs = uvecs[mask_i]
            n_b = len(vecs)
            if n_b == 0:
                continue

            theta = np.arccos(np.clip(vecs[:, 2], -1.0, 1.0))
            phi = np.arctan2(vecs[:, 1], vecs[:, 0])
            phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)

            sum_sq = 0.0
            for m in range(-l, l + 1):
                ylm = sph_harm_y(l, m, theta, phi)
                q_lm_i = ylm.mean()
                sum_sq += q_lm_i.real ** 2 + q_lm_i.imag ** 2

            q_l_vals[i] = float(np.sqrt(prefactor * sum_sq))

        return q_l_vals

    def steinhardt_wl(
        self,
        l: int,
        cutoff: float,
        neighbor_elements: list[str] | None = None,
    ) -> np.ndarray:
        """Third-order rotational invariant w_l per atom.

        w_l = sum_{m1+m2+m3=0} W3j(l,m1;l,m2;l,m3) * q_lm1 * q_lm2 * q_lm3
              / (sum_m |q_lm|^2)^(3/2)

        Wigner 3j symbols are precomputed once via sympy.

        Parameters
        ----------
        l : int
            Degree (e.g. 4 or 6).
        cutoff : float
            Neighbor cutoff distance (Angstrom).
        neighbor_elements : list of str or None
            If given, only neighbors in this list contribute.

        Returns
        -------
        w_l : np.ndarray, shape (N_atoms,)
            Normalised w_l per atom.
        """
        from scipy.special import sph_harm_y
        from sympy.physics.wigner import wigner_3j

        # Precompute nonzero Wigner 3j values for this l
        w3j_cache: dict[tuple[int, int, int], float] = {}
        for m1 in range(-l, l + 1):
            for m2 in range(-l, l + 1):
                m3 = -(m1 + m2)
                if abs(m3) <= l:
                    val = float(wigner_3j(l, l, l, m1, m2, m3))
                    if val != 0.0:
                        w3j_cache[(m1, m2, m3)] = val

        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())

        i_arr, j_arr, D_arr, dist_arr = self._neighbors_vesin(cutoff)

        if neighbor_elements is not None:
            ne_set = set(neighbor_elements)
            elem_mask = np.array([symbols[j] in ne_set for j in j_arr])
            i_arr, j_arr, D_arr, dist_arr = (
                i_arr[elem_mask], j_arr[elem_mask],
                D_arr[elem_mask], dist_arr[elem_mask],
            )

        valid = dist_arr > 0
        i_arr, j_arr, D_arr, dist_arr = (
            i_arr[valid], j_arr[valid], D_arr[valid], dist_arr[valid]
        )
        uvecs = D_arr / dist_arr[:, None]

        w_l_vals = np.zeros(len(atoms), dtype=float)

        for i in range(len(atoms)):
            mask_i = (i_arr == i)
            vecs = uvecs[mask_i]
            n_b = len(vecs)
            if n_b == 0:
                continue

            theta = np.arccos(np.clip(vecs[:, 2], -1.0, 1.0))
            phi = np.arctan2(vecs[:, 1], vecs[:, 0])
            phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)

            # Compute q_lm for all m
            q_lm: dict[int, complex] = {}
            sum_sq = 0.0
            for m in range(-l, l + 1):
                ylm = sph_harm_y(l, m, theta, phi)
                q_lm_i = complex(ylm.mean())
                q_lm[m] = q_lm_i
                sum_sq += q_lm_i.real ** 2 + q_lm_i.imag ** 2

            # Numerator: sum over nonzero Wigner 3j terms
            numerator = 0.0 + 0.0j
            for (m1, m2, m3), w3j_val in w3j_cache.items():
                numerator += w3j_val * q_lm[m1] * q_lm[m2] * q_lm[m3]

            denom = (sum_sq + 1e-10) ** 1.5
            w_l_vals[i] = float(numerator.real / denom)

        return w_l_vals

    # ------------------------------------------------------------------
    # Ring statistics
    # ------------------------------------------------------------------

    def ring_statistics(
        self,
        element1: str,
        element2: str,
        cutoff: float,
        max_ring: int = 10,
    ) -> dict[int, int]:
        """
        Count rings (of size 3..max_ring) in the element1-element2 bond network.

        Only rings that alternate between element1 and element2 are counted
        (e.g. Ta-Cl-Ta-Cl rings for octahedral networks).

        Returns
        -------
        dict {ring_size: count}
        """
        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())
        elem_set = {element1, element2}

        i_arr, j_arr, _D, dist_arr = self._neighbors_vesin(cutoff)

        # Build undirected graph — nodes are atoms, edges are bonds
        G = nx.Graph()
        for idx in range(len(atoms)):
            if symbols[idx] in elem_set:
                G.add_node(idx, element=symbols[idx])

        valid = (dist_arr > 0)
        for ii, jj in zip(i_arr[valid], j_arr[valid]):
            if symbols[ii] in elem_set and symbols[jj] in elem_set:
                G.add_edge(int(ii), int(jj))

        ring_counts = {size: 0 for size in range(3, max_ring + 1)}

        # Use minimum_cycle_basis for efficiency
        try:
            cycles = nx.minimum_cycle_basis(G)
        except Exception:
            warnings.warn(
                "ring_statistics: networkx cycle basis failed; returning empty dict"
            )
            return ring_counts

        for cycle in cycles:
            size = len(cycle)
            if 3 <= size <= max_ring:
                ring_counts[size] += 1

        return ring_counts

    # ------------------------------------------------------------------
    # F_IS grouped by coordination number
    # ------------------------------------------------------------------

    def fis_by_cn(
        self,
        central: str,
        neighbor: str,
        cutoff: float,
        central_z: int,
        neighbor_z: int | None = None,
    ) -> dict[int, np.ndarray]:
        """Return F_IS per atom grouped by their coordination number.

        Useful to validate that CN=4 → F_IS≈−1/3 and CN=6 → F_IS≈+1.

        Returns
        -------
        dict mapping CN (int) → np.ndarray of F_IS values for atoms at that CN.
        """
        # Compute coordination numbers
        cn_arr = self.coordination_numbers(central, neighbor, cutoff)

        # Compute F_IS for central atoms
        op = self.order_params_per_atom(
            central_z=central_z,
            neighbor_z=neighbor_z,
            cutoff=cutoff,
            compute=["fis"],
        )
        fis_arr = op.get("fis", np.array([]))
        if len(fis_arr) == 0 or len(cn_arr) == 0:
            return {}

        result: dict[int, np.ndarray] = {}
        for cn_val in np.unique(cn_arr):
            mask = cn_arr == cn_val
            result[int(cn_val)] = fis_arr[mask]
        return result

    # ------------------------------------------------------------------
    # Polyhedral distortion index
    # ------------------------------------------------------------------

    def polyhedral_distortion_index(
        self,
        central: str,
        neighbor: str,
        cutoff: float,
    ) -> np.ndarray:
        """Bond-length distortion index DI per central atom.

        DI_i = (1/N_b) * Σ_j |d_ij − d̄_i| / d̄_i

        DI = 0 for a perfect regular polyhedron; larger values indicate
        distorted environments (relevant for Fe2O3 tet vs oct mixing,
        NaTaCl6 octahedral distortion, LiPON N-substitution distortion).

        Returns
        -------
        di : np.ndarray, shape (n_central_atoms,)
        """
        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())
        idx_c = np.where(symbols == central)[0]
        idx_n = np.where(symbols == neighbor)[0]

        if len(idx_c) == 0 or len(idx_n) == 0:
            return np.array([])

        i_arr, j_arr, _D, dist_arr = self._neighbors_vesin(cutoff)
        mask = np.isin(i_arr, idx_c) & np.isin(j_arr, idx_n) & (dist_arr > 0)
        i_f, dist_f = i_arr[mask], dist_arr[mask]

        # Map global atom index → position in idx_c
        central_pos = {int(a): k for k, a in enumerate(idx_c)}

        di = np.zeros(len(idx_c), dtype=float)
        for k, ci in enumerate(idx_c):
            bond_dists = dist_f[i_f == ci]
            if len(bond_dists) == 0:
                continue
            d_mean = bond_dists.mean()
            if d_mean > 0:
                di[k] = np.abs(bond_dists - d_mean).mean() / d_mean
        return di

    # ------------------------------------------------------------------
    # Warren-Cowley short-range order parameters
    # ------------------------------------------------------------------

    def warren_cowley_sro(
        self,
        cutoff: float,
        elements: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Warren-Cowley SRO parameters α_ij = 1 − P(j|i) / x_j.

        α_ij > 0 : species i avoids j (like-atom clustering)
        α_ij < 0 : species i prefers j (chemical ordering / anti-clustering)
        α_ij = 0 : random mixing

        Parameters
        ----------
        cutoff : float
            First-shell cutoff distance (Å).
        elements : list of str or None
            Subset of species to include. None = all species in the structure.

        Returns
        -------
        alpha : np.ndarray, shape (n_elem, n_elem)
        elem_list : list[str]
            Ordered element labels matching alpha rows/cols.
        """
        atoms = self.atoms
        syms = np.array(atoms.get_chemical_symbols())
        all_elems = sorted(set(syms.tolist()))
        if elements is not None:
            all_elems = [e for e in all_elems if e in elements]
        elem_idx = {e: k for k, e in enumerate(all_elems)}
        n_e = len(all_elems)

        # Mole fractions
        x = np.zeros(n_e)
        for e, k in elem_idx.items():
            x[k] = (syms == e).sum() / len(syms)

        i_arr, j_arr, _D, dist_arr = self._neighbors_vesin(cutoff)
        valid = dist_arr > 0
        i_arr, j_arr = i_arr[valid], j_arr[valid]

        # N_ij = number of j-neighbours around i-type atoms
        N_ij = np.zeros((n_e, n_e), dtype=float)
        N_i  = np.zeros(n_e, dtype=float)  # total neighbours of i-type atoms

        for ii, jj in zip(i_arr, j_arr):
            si, sj = syms[ii], syms[jj]
            if si not in elem_idx or sj not in elem_idx:
                continue
            ki, kj = elem_idx[si], elem_idx[sj]
            N_ij[ki, kj] += 1
            N_i[ki] += 1

        alpha = np.zeros((n_e, n_e), dtype=float)
        for ki in range(n_e):
            for kj in range(n_e):
                if N_i[ki] > 0 and x[kj] > 0:
                    P_ji = N_ij[ki, kj] / N_i[ki]
                    alpha[ki, kj] = 1.0 - P_ji / x[kj]
        return alpha, all_elems

    # ------------------------------------------------------------------
    # F_IS spatial autocorrelation
    # ------------------------------------------------------------------

    def fis_spatial_autocorrelation(
        self,
        central: str,
        neighbor: str,
        cutoff: float,
        central_z: int,
        neighbor_z: int | None,
        r_max: float = 8.0,
        n_bins: int = 40,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spatial pair-correlation of the F_IS field.

        C(r) = <δF_IS(i) δF_IS(j)> / <δF_IS²>

        averaged over all pairs (i,j) separated by r ± Δr/2, where
        δF_IS = F_IS − <F_IS>.

        C(r)≈1 at short range means neighbouring atoms share the same local
        symmetry character (e.g. both tetrahedral or both octahedral).
        C(r)≈0 means uncorrelated beyond that distance.

        Returns
        -------
        r_centers : np.ndarray, shape (n_bins,)
        C_r       : np.ndarray, shape (n_bins,)
        """
        # F_IS values for central atoms only
        op = self.order_params_per_atom(
            central_z=central_z,
            neighbor_z=neighbor_z,
            cutoff=cutoff,
            compute=["fis"],
        )
        fis_all = op.get("fis", np.array([]))
        if len(fis_all) == 0:
            r_edges = np.linspace(0, r_max, n_bins + 1)
            return 0.5 * (r_edges[:-1] + r_edges[1:]), np.zeros(n_bins)

        atoms = self.atoms
        symbols = np.array(atoms.get_chemical_symbols())
        idx_c = np.where(symbols == central)[0]

        # Build per-atom F_IS array indexed by global atom index
        fis_global = np.full(len(atoms), np.nan)
        fis_global[idx_c] = fis_all

        delta_fis = fis_all - fis_all.mean()
        var_fis = (delta_fis ** 2).mean()
        if var_fis < 1e-12:
            r_edges = np.linspace(0, r_max, n_bins + 1)
            return 0.5 * (r_edges[:-1] + r_edges[1:]), np.zeros(n_bins)

        # Pair distances among central atoms only
        i_arr, j_arr, _D, dist_arr = self._neighbors_vesin(r_max)
        mask = (
            np.isin(i_arr, idx_c) & np.isin(j_arr, idx_c) & (dist_arr > 0)
        )
        i_f, j_f, dist_f = i_arr[mask], j_arr[mask], dist_arr[mask]

        # Map global idx → position in fis_all
        g2l = {int(a): k for k, a in enumerate(idx_c)}

        bin_edges = np.linspace(0, r_max, n_bins + 1)
        r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        numerator = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)

        bin_ids = np.digitize(dist_f, bin_edges) - 1
        valid_bin = (bin_ids >= 0) & (bin_ids < n_bins)

        for k in range(len(i_f)):
            if not valid_bin[k]:
                continue
            gi, gj = int(i_f[k]), int(j_f[k])
            if gi not in g2l or gj not in g2l:
                continue
            li, lj = g2l[gi], g2l[gj]
            b = bin_ids[k]
            numerator[b] += delta_fis[li] * delta_fis[lj]
            counts[b] += 1

        denom = counts * var_fis
        C_r = np.where(counts > 0, np.divide(numerator, denom, where=counts > 0, out=np.zeros_like(numerator)), 0.0)
        return r_centers, C_r
