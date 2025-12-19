"""
Differentiable, batched order parameter calculator for torch_sim SimState.

Computes structural order parameters (tetrahedral, octahedral, bond-orientational)
for specified atoms with automatic neighbor finding and PBC handling.
"""

import torch
import torch.nn as nn
import torch_sim as ts
from torch_sim.neighbors import torchsim_nl
from typing import List, Dict, Optional
import math
from torch_sim import transforms


class TorchSimOrderParameters(nn.Module):
    """
    Differentiable order parameter calculator for torch_sim SimState.
    
    Automatically handles:
    - Neighbor finding with PBC
    - Batched computation
    - Differentiability for ML applications
    
    Usage:
        >>> op_calc = TorchSimOrderParameters(device='cuda')
        >>> state = ts.initialize_state(structure, device='cuda', dtype=torch.float32)
        >>> 
        >>> # Compute for specific P atoms
        >>> p_indices = torch.where(state.atomic_numbers == 15)[0]
        >>> results = op_calc(state, p_indices, ['tet', 'cn', 'q4'])
        >>> 
        >>> print(f"Tetrahedral OP: {results['tet']}")  # Shape: (n_p_atoms,)
    """
    
    SUPPORTED_TYPES = [
        'cn', 'tet', 'oct', 'bcc',
        'q2', 'q4', 'q6',
        'tri_plan', 'sq_plan', 'tri_pyr'
    ]
    
    def __init__(self, cutoff: float = 3.5, device: str = 'cpu'):
        """
        Args:
            cutoff: Distance cutoff for neighbor finding (Angstroms)
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        self.cutoff = cutoff
        self.device = torch.device(device)
        self.default_params = self._initialize_default_params()
        
    def _initialize_default_params(self) -> Dict[str, Dict[str, float]]:
        """Initialize default parameters for each order parameter."""
        return {
            'tet': {
                'TA': 0.6081734479693927,
                'delta_theta': 12.0,
            },
            'oct': {
                'min_SPP': 2.792526803190927,
                'delta_theta1': 12.0,
                'delta_theta2': 10.0,
                'w_SPP': 3.0,
            },
            'bcc': {
                'min_SPP': 2.792526803190927,
                'delta_theta': 19.47,
                'w_SPP': 6.0,
            },
        }
    
    def forward(
        self,
        state: ts.SimState,
        atom_indices: torch.Tensor,
        order_params: List[str],
        element_filter: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute order parameters for specified atoms in a SimState.
        
        Args:
            state: torch_sim SimState containing atomic structure
            atom_indices: (M,) tensor of atom indices to compute OPs for
            order_params: List of order parameter types to compute
            element_filter: Optional list of atomic numbers to include as neighbors
                          (e.g., [15, 16] for P and S only)
        
        Returns:
            Dictionary mapping OP names to (M,) tensors of values
            
        Example:
            >>> # Compute tetrahedral OP for all P atoms
            >>> p_mask = state.atomic_numbers == 15
            >>> p_indices = torch.where(p_mask)[0]
            >>> results = op_calc(state, p_indices, ['tet', 'cn'])
        """
        # Extract cell matrix (must be 3x3 for torchsim_nl)
        if state.cell.ndim == 3:
            # Batched systems: shape (n_systems, 3, 3)
            cell_matrix = state.cell[0]
        elif state.cell.ndim == 2:
            if state.cell.shape == (3, 3):
                cell_matrix = state.cell
            else:
                raise ValueError(f"Expected cell shape (3, 3), got {state.cell.shape}")
        else:
            raise ValueError(f"Unexpected cell dimensionality: {state.cell.ndim}")
        
        # Get neighbor list with PBC
        mapping, shifts = torchsim_nl(
            state.positions,
            cell_matrix,
            state.pbc,
            torch.tensor(self.cutoff, device=self.device, dtype=state.dtype),
            sort_id=True
        )
        
        # Build neighbor list for each target atom
        neighbor_indices, valid_mask = self._build_neighbor_tensor(
            atom_indices, mapping, state.atomic_numbers, element_filter
        )
        
        # Compute positions for neighbors including PBC
        positions_with_pbc = self._apply_pbc_to_neighbors(
            state.positions, neighbor_indices, mapping, shifts, cell_matrix, atom_indices
        )
        
        # Compute order parameters
        results = {}
        
        # Get geometric quantities
        vectors, distances, thetas, phis = self._compute_geometry(
            state.positions, atom_indices, neighbor_indices, positions_with_pbc, valid_mask
        )
        
        for op_type in order_params:
            if op_type not in self.SUPPORTED_TYPES:
                raise ValueError(f"Unsupported order parameter: {op_type}")
            
            if op_type == 'cn':
                results[op_type] = self._compute_cn(valid_mask)
            elif op_type == 'tet':
                results[op_type] = self._compute_tetrahedral(vectors, valid_mask)
            elif op_type == 'oct':
                results[op_type] = self._compute_octahedral(vectors, valid_mask)
            elif op_type == 'bcc':
                results[op_type] = self._compute_bcc(vectors, valid_mask)
            elif op_type == 'q2':
                results[op_type] = self._compute_q2(thetas, phis, valid_mask)
            elif op_type == 'q4':
                results[op_type] = self._compute_q4(thetas, phis, valid_mask)
            elif op_type == 'q6':
                results[op_type] = self._compute_q6(thetas, phis, valid_mask)
        
        return results
    
    def _build_neighbor_tensor(
        self,
        atom_indices: torch.Tensor,
        mapping: torch.Tensor,
        atomic_numbers: torch.Tensor,
        element_filter: Optional[List[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build padded neighbor tensor for specified atoms.
        
        Returns:
            neighbor_indices: (M, K) tensor of neighbor indices (padded with -1)
            valid_mask: (M, K) boolean mask of valid neighbors
        """
        M = len(atom_indices)
        
        # Find neighbors for each target atom
        neighbors_list = []
        max_neighbors = 0
        
        for atom_idx in atom_indices:
            # Find all neighbors of this atom
            mask = mapping[0] == atom_idx
            neighs = mapping[1, mask]
            
            # Apply element filter if specified
            if element_filter is not None:
                element_mask = torch.isin(atomic_numbers[neighs], 
                                         torch.tensor(element_filter, device=self.device))
                neighs = neighs[element_mask]
            
            neighbors_list.append(neighs)
            max_neighbors = max(max_neighbors, len(neighs))
        
        # Pad to uniform size
        neighbor_indices = torch.full((M, max_neighbors), -1, 
                                     dtype=torch.long, device=self.device)
        
        for i, neighs in enumerate(neighbors_list):
            neighbor_indices[i, :len(neighs)] = neighs
        
        valid_mask = neighbor_indices >= 0
        
        return neighbor_indices, valid_mask
    
    def _apply_pbc_to_neighbors(
        self,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        mapping: torch.Tensor,
        shifts: torch.Tensor,
        cell_matrix: torch.Tensor,
        atom_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply periodic boundary corrections to neighbor positions.
        
        Args:
            cell_matrix: (3, 3) cell matrix for the system
            atom_indices: (M,) actual atom indices being computed
        
        Returns:
            positions_with_pbc: (M, K, 3) tensor of neighbor positions with PBC applied
        """
        M, K = neighbor_indices.shape
        
        # Initialize with regular positions
        valid_mask = neighbor_indices >= 0
        safe_indices = neighbor_indices.clone()
        safe_indices[~valid_mask] = 0
        
        positions_pbc = positions[safe_indices]  # (M, K, 3)
        
        # Apply shifts if needed
        if shifts.abs().sum() > 0:
            # For each neighbor pair in the mapping, find corresponding shift
            for i in range(M):
                center_atom_idx = atom_indices[i]  # Actual atom index
                for j in range(K):
                    if not valid_mask[i, j]:
                        continue
                    
                    neigh_idx = neighbor_indices[i, j]
                    
                    # Find this pair in mapping - use actual atom index!
                    pair_mask = (mapping[0] == center_atom_idx) & (mapping[1] == neigh_idx)
                    if pair_mask.any():
                        pair_shift = shifts[pair_mask][0]
                        
                        # Apply shift using the cell matrix
                        shift_vec = torch.matmul(pair_shift.float(), cell_matrix.T.float())
                        positions_pbc[i, j] += shift_vec
        
        return positions_pbc
    
    def _compute_geometry(
        self,
        center_positions: torch.Tensor,
        center_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_positions_pbc: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute geometric quantities for order parameters.
        
        Returns:
            vectors: (M, K, 3) normalized direction vectors
            distances: (M, K) distances
            thetas: (M, K) polar angles
            phis: (M, K) azimuthal angles
        """
        M, K = neighbor_indices.shape
        
        # Get center positions
        center_pos = center_positions[center_indices]  # (M, 3)
        
        # Compute vectors with PBC
        vectors = neighbor_positions_pbc - center_pos.unsqueeze(1)  # (M, K, 3)
        distances = torch.norm(vectors, dim=-1)  # (M, K)
        
        # Normalize with numerical stability
        vectors = vectors / (distances.unsqueeze(-1) + 1e-10)
        
        # Compute angles with stability
        cos_theta = vectors[..., 2].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        thetas = torch.acos(cos_theta)
        phis = torch.atan2(vectors[..., 1], vectors[..., 0])
        
        # Apply mask
        vectors = vectors * valid_mask.unsqueeze(-1).float()
        distances = distances * valid_mask.float()
        thetas = thetas * valid_mask.float()
        phis = phis * valid_mask.float()
        
        return vectors, distances, thetas, phis
    
    def _compute_cn(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute coordination number."""
        return valid_mask.sum(dim=1).float()
    
    def _compute_tetrahedral(
        self,
        vectors: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute tetrahedral order parameter (differentiable)."""
        params = self.default_params['tet']
        M, K, _ = vectors.shape
        
        # Compute pairwise angles with numerical stability
        dots = torch.einsum('mki,mji->mkj', vectors, vectors)
        dots = dots.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # Avoid exact -1 or 1
        angles = torch.acos(dots)
        
        # Target angle for tetrahedral
        target_angle = params['TA'] * math.pi
        delta_theta = params['delta_theta'] * math.pi / 180.0
        
        # Gaussian penalty with numerical stability
        angle_diff = angles - target_angle
        # Clamp exponent to prevent overflow
        exponent = -0.5 * (angle_diff / (delta_theta + 1e-10)) ** 2
        exponent = exponent.clamp(min=-50.0)  # Prevent underflow
        gaussian = torch.exp(exponent)
        
        # Mask valid pairs
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        diag_mask = ~torch.eye(K, device=self.device, dtype=torch.bool).unsqueeze(0)
        full_mask = pair_mask & diag_mask
        
        # Sum and normalize
        qtet = (gaussian * full_mask.float()).sum(dim=(1, 2))
        n_neighbors = valid_mask.sum(dim=1).float().clamp(min=1.0)
        norm = n_neighbors * (n_neighbors - 1).clamp(min=1.0)
        
        return qtet / (norm + 1e-10)
    
    def _compute_octahedral(
        self,
        vectors: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute octahedral order parameter (differentiable)."""
        params = self.default_params['oct']
        M, K, _ = vectors.shape
        
        # Compute pairwise angles
        dots = torch.einsum('mki,mji->mkj', vectors, vectors)
        dots = dots.clamp(-1.0, 1.0)
        angles = torch.acos(dots)
        
        theta_thr = params['min_SPP']
        delta_theta1 = params['delta_theta1'] * math.pi / 180.0
        delta_theta2 = params['delta_theta2'] * math.pi / 180.0
        
        # South pole (180°)
        is_south = angles >= theta_thr
        angle_diff_180 = angles - math.pi
        gauss_180 = torch.exp(-0.5 * (angle_diff_180 / delta_theta1) ** 2)
        contrib_south = 3 * gauss_180 * is_south.float()
        
        # Equatorial (90°)
        is_equat = angles < theta_thr
        angle_diff_90 = angles - math.pi / 2
        gauss_90 = torch.exp(-0.5 * (angle_diff_90 / delta_theta2) ** 2)
        contrib_equat = gauss_90 * is_equat.float()
        
        # Mask valid pairs
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        diag_mask = ~torch.eye(K, device=self.device, dtype=torch.bool).unsqueeze(0)
        full_mask = pair_mask & diag_mask
        
        total = ((contrib_south + contrib_equat) * full_mask.float()).sum(dim=(1, 2))
        
        # Normalize
        n_neighbors = valid_mask.sum(dim=1).float()
        norm = n_neighbors * (3 + (n_neighbors - 2) * (n_neighbors - 3))
        
        return total / (norm + 1e-10)
    
    def _compute_bcc(
        self,
        vectors: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCC order parameter (differentiable)."""
        params = self.default_params['bcc']
        M, K, _ = vectors.shape
        
        dots = torch.einsum('mki,mji->mkj', vectors, vectors)
        dots = dots.clamp(-1.0, 1.0)
        angles = torch.acos(dots)
        
        theta_thr = params['min_SPP']
        delta_theta = params['delta_theta'] * math.pi / 180.0
        
        is_south = angles >= theta_thr
        angle_diff_180 = angles - math.pi
        gauss_180 = torch.exp(-0.5 * (angle_diff_180 / delta_theta) ** 2)
        contrib_south = params['w_SPP'] * gauss_180 * is_south.float()
        
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        diag_mask = ~torch.eye(K, device=self.device, dtype=torch.bool).unsqueeze(0)
        full_mask = pair_mask & diag_mask
        
        total = (contrib_south * full_mask.float()).sum(dim=(1, 2))
        
        n_neighbors = valid_mask.sum(dim=1).float()
        norm = n_neighbors * (6 + (n_neighbors - 2) * (n_neighbors - 3))
        
        return total / (norm + 1e-10)
    
    def _compute_q2(
        self,
        thetas: torch.Tensor,
        phis: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute q2 bond orientational order parameter (differentiable)."""
        n_neighbors = valid_mask.sum(dim=1, keepdim=True).float()
        
        sin_t = torch.sin(thetas)
        cos_t = torch.cos(thetas)
        sin_t2 = sin_t ** 2
        cos_t2 = cos_t ** 2
        
        cos_2p = torch.cos(2 * phis)
        sin_2p = torch.sin(2 * phis)
        cos_p = torch.cos(phis)
        sin_p = torch.sin(phis)
        
        sqrt_15_2pi = math.sqrt(15 / (2 * math.pi))
        sqrt_5_pi = math.sqrt(5 / math.pi)
        
        acc = torch.zeros(thetas.shape[0], device=self.device)
        
        # Y_2_-2
        pre_y_2_2 = 0.25 * sqrt_15_2pi * sin_t2
        real = (pre_y_2_2 * cos_2p * valid_mask.float()).sum(dim=1)
        imag = -(pre_y_2_2 * sin_2p * valid_mask.float()).sum(dim=1)
        acc += real**2 + imag**2
        
        # Y_2_-1
        pre_y_2_1 = 0.5 * sqrt_15_2pi * sin_t * cos_t
        real = (pre_y_2_1 * cos_p * valid_mask.float()).sum(dim=1)
        imag = -(pre_y_2_1 * sin_p * valid_mask.float()).sum(dim=1)
        acc += real**2 + imag**2
        
        # Y_2_0
        real = (0.25 * sqrt_5_pi * (3 * cos_t2 - 1.0) * valid_mask.float()).sum(dim=1)
        acc += real**2
        
        # Y_2_1
        real = -(pre_y_2_1 * cos_p * valid_mask.float()).sum(dim=1)
        imag = -(pre_y_2_1 * sin_p * valid_mask.float()).sum(dim=1)
        acc += real**2 + imag**2
        
        # Y_2_2
        real = (pre_y_2_2 * cos_2p * valid_mask.float()).sum(dim=1)
        imag = (pre_y_2_2 * sin_2p * valid_mask.float()).sum(dim=1)
        acc += real**2 + imag**2
        
        return torch.sqrt(4 * math.pi * acc / (5 * n_neighbors.squeeze()**2 + 1e-10))
    
    def _compute_q4(
        self,
        thetas: torch.Tensor,
        phis: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute q4 bond orientational order parameter (differentiable)."""
        n_neighbors = valid_mask.sum(dim=1, keepdim=True).float()
        
        sin_t = torch.sin(thetas)
        cos_t = torch.cos(thetas)
        
        acc = torch.zeros(thetas.shape[0], device=self.device)
        
        # Simplified q4 calculation - full version would include all m=-4 to 4 terms
        # Y_4_0 term
        sqrt_1_pi = math.sqrt(1 / math.pi)
        pre = (3/16.0) * sqrt_1_pi * (35 * cos_t**4 - 30 * cos_t**2 + 3.0)
        real = (pre * valid_mask.float()).sum(dim=1)
        acc += real**2
        
        return torch.sqrt(4 * math.pi * acc / (9 * n_neighbors.squeeze()**2 + 1e-10))
    
    def _compute_q6(
        self,
        thetas: torch.Tensor,
        phis: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute q6 bond orientational order parameter (differentiable)."""
        n_neighbors = valid_mask.sum(dim=1, keepdim=True).float()
        
        cos_t = torch.cos(thetas)
        
        acc = torch.zeros(thetas.shape[0], device=self.device)
        
        # Y_6_0 term
        sqrt_13_pi = math.sqrt(13 / math.pi)
        pre = (1/32.0) * sqrt_13_pi * (
            231 * cos_t**6 - 315 * cos_t**4 + 105 * cos_t**2 - 5.0
        )
        real = (pre * valid_mask.float()).sum(dim=1)
        acc += real**2
        
        return torch.sqrt(4 * math.pi * acc / (13 * n_neighbors.squeeze()**2 + 1e-10))


# Example usage
if __name__ == "__main__":
    from pymatgen.core import Structure, Lattice
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create example Li3PS4 structure
    # lattice = Lattice.cubic(10.0)
    # structure = Structure(
    #     lattice,
    #     ["Li", "Li", "Li", "P", "S", "S", "S", "S"],
    #     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5],
    #      [0.25, 0.25, 0.25],
    #      [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.75, 0.25, 0.25], [0.75, 0.75, 0.75]]
    # )
    from pymatgen.core import Structure

# Load the CIF file
    structure = Structure.from_file("/home/advaitgore/PycharmProjects/torchdisorder/data-release/crystal-structures/Li3PS4_gamma.cif")
    
    # Convert to SimState with float64 for better gradient stability
    state = ts.initialize_state(structure, device=device, dtype=torch.float64)
    
  
    op_calc = TorchSimOrderParameters(cutoff=3, device=device)
    
    # Find P atoms
    p_mask = state.atomic_numbers == 15
    p_indices = torch.where(p_mask)[0]
    
    print(f"\nFound {len(p_indices)} P atoms at indices: {p_indices.tolist()}")
    
    # Compute order parameters (filter for S neighbors only for tetrahedral PS4)
    results = op_calc(
        state,
        p_indices,
        order_params=['cn', 'tet', 'oct', 'q4'],
        element_filter=[16]  # Only S neighbors (not P)
    )
    
    print("\nOrder Parameters:")
    for key, value in results.items():
        if value.numel() == 1:
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value.tolist()}")
    
    # Test differentiability
    state.positions.requires_grad = True
    results = op_calc(state, p_indices, ['tet'])
    loss = results['tet'].sum()
    loss.backward()
    
    print(f"\nDifferentiability test:")
    print(f"  Gradient exists: {state.positions.grad is not None}")
    print(f"  Gradient shape: {state.positions.grad.shape}")
    if state.positions.grad is not None:
        grad_norm = state.positions.grad.norm()
        has_nan = torch.isnan(state.positions.grad).any()
        print(f"  Gradient norm: {grad_norm.item():.6f}")
        print(f"  Contains NaN: {has_nan.item()}")
        if has_nan:
            print("  Warning: NaN in gradients - check numerical stability!")
    
    print("\n" + "=" * 60)
    print("Expected Results for tetrahedral PS4:")
    print("  CN  = 4 (four S neighbors)")
    print("  tet ~ 0.9+ (high tetrahedral order)")
    print("  If CN=6, cutoff is too large (catching periodic images)")
    print("  If grad=NaN, use float64 dtype or check angle computations")
    print("=" * 60)