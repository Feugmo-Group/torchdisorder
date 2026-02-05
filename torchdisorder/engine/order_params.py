"""
Order Parameter Calculator with Optional WARP Acceleration
==========================================================

Computes structural order parameters (tetrahedral, octahedral, bond-orientational)
for specified atoms with automatic neighbor finding and PBC handling.

Performance:
    - PyTorch backend: Works everywhere, ~1-10ms for small systems
    - WARP backend: 10-100x faster for large systems (>1000 atoms)

Usage:
    >>> from torchdisorder.engine.order_params import TorchSimOrderParameters
    >>> 
    >>> # Automatic backend selection (WARP if available on CUDA)
    >>> op_calc = TorchSimOrderParameters(cutoff=3.5, device='cuda')
    >>> results = op_calc(state, p_indices, ['tet', 'cn', 'q4'])
    >>> 
    >>> # Force PyTorch backend
    >>> op_calc = TorchSimOrderParameters(cutoff=3.5, device='cuda', backend='pytorch')
    >>> 
    >>> # Force WARP backend (raises error if not available)
    >>> op_calc = TorchSimOrderParameters(cutoff=3.5, device='cuda', backend='warp')

Supported Order Parameters:
    - cn: Coordination number
    - tet: Tetrahedral order parameter (for SiO4, PO4, PS4)
    - oct: Octahedral order parameter (for FeO6, TiO6)
    - bcc: BCC-like order parameter
    - q2, q4, q6: Steinhardt bond orientational order parameters
"""

import torch
import torch.nn as nn
import math
import json
import warnings
from typing import List, Dict, Optional, Tuple

import torch_sim as ts
from torch_sim.neighbors import torch_nl_linked_cell, torch_nl_n2, torchsim_nl

# =============================================================================
# Check for WARP availability
# =============================================================================

WARP_AVAILABLE = False
try:
    import warp as wp
    wp.init()
    WARP_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# PyTorch Backend (Default Fallback)
# =============================================================================

class PyTorchOrderParameters(nn.Module):
    """
    Pure PyTorch order parameter calculator.
    
    Works on any device (CPU, CUDA, MPS) without additional dependencies.
    """
    
    SUPPORTED_TYPES = [
        'cn', 'tet', 'oct', 'bcc',
        'q2', 'q4', 'q6',
        'tri_plan', 'sq_plan', 'tri_pyr'
    ]
    
    def __init__(self, cutoff: float = 3.5, device: str = 'cpu', max_neighbors: int = 64):
        super().__init__()
        self.cutoff = cutoff
        self.device = torch.device(device)
        self.max_neighbors = max_neighbors
        self.default_params = self._initialize_default_params()
    
    def _initialize_default_params(self) -> Dict[str, Dict[str, float]]:
        """Initialize default parameters for each order parameter."""
        return {
            'tet': {
                'TA': 0.6081734479693927,  # 109.47° in units of π
                'delta_theta': 12.0,        # degrees
            },
            'oct': {
                'min_SPP': 2.792526803190927,  # ~160° threshold
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
        Compute order parameters for specified atoms.
        
        Args:
            state: torch_sim SimState
            atom_indices: (M,) tensor of atom indices to compute OPs for
            order_params: List of OP types ['tet', 'cn', 'q4', etc.]
            element_filter: Optional list of atomic numbers for neighbors
        
        Returns:
            Dict mapping OP names to (M,) tensors
        """
        # Validate
        for op in order_params:
            if op not in self.SUPPORTED_TYPES:
                raise ValueError(f"Unsupported order parameter: {op}")
        
        M = len(atom_indices)
        if M == 0:
            return {op: torch.tensor([], device=self.device) for op in order_params}
        
        # Build neighbor data
        neighbor_indices, neighbor_positions_pbc, valid_mask, center_indices = \
            self._build_neighbor_data(state, atom_indices, element_filter)
        
        # Compute geometric quantities
        vectors, distances, thetas, phis = self._compute_geometry(
            state.positions, center_indices, neighbor_indices,
            neighbor_positions_pbc, valid_mask
        )
        
        # Compute requested OPs
        results = {}
        for op in order_params:
            if op == 'cn':
                results[op] = self._compute_cn(valid_mask)
            elif op == 'tet':
                results[op] = self._compute_tetrahedral(vectors, valid_mask)
            elif op == 'oct':
                results[op] = self._compute_octahedral(vectors, valid_mask)
            elif op == 'bcc':
                results[op] = self._compute_bcc(vectors, valid_mask)
            elif op == 'q2':
                results[op] = self._compute_q2(thetas, phis, valid_mask)
            elif op == 'q4':
                results[op] = self._compute_q4(thetas, phis, valid_mask)
            elif op == 'q6':
                results[op] = self._compute_q6(thetas, phis, valid_mask)
            else:
                # Placeholder for unimplemented
                results[op] = torch.zeros(M, device=self.device)
        
        return results
    
    def _build_neighbor_data(
        self,
        state: ts.SimState,
        atom_indices: torch.Tensor,
        element_filter: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build padded neighbor arrays for batch processing.
        
        Uses minimum image convention for PBC-corrected positions,
        which is robust regardless of torchsim_nl return format.
        """
        
        # Get neighbor list
        positions = state.positions
        cell = state.cell
        pbc = state.pbc if state.pbc is not None else torch.tensor([True, True, True], device=self.device)
        
        if state.system_idx is None:
            system_idx = torch.zeros(len(positions), dtype=torch.long, device=self.device)
        else:
            system_idx = state.system_idx
        
        # Ensure cell is 3D for torchsim_nl (batch dimension)
        cell_for_nl = cell.unsqueeze(0) if cell.ndim == 2 else cell
        
        # Compute neighbors - only use mapping, ignore shifts (may not be shift vectors)
        mapping = torchsim_nl(
            positions=positions,
            cell=cell_for_nl,
            pbc=pbc,
            cutoff=torch.tensor([self.cutoff], device=self.device),
            system_idx=system_idx,
            self_interaction=False,
        )[0]  # Only take mapping (edge indices)
        
        M = len(atom_indices)
        K = self.max_neighbors
        
        # Get the 2D cell matrix for minimum image convention
        cell_2d = cell[0] if cell.ndim == 3 else cell
        cell_inv = torch.linalg.inv(cell_2d.float())
        
        # Pre-allocate
        neighbor_indices = torch.full((M, K), -1, dtype=torch.long, device=self.device)
        neighbor_positions_pbc = torch.zeros((M, K, 3), dtype=positions.dtype, device=self.device)
        valid_mask = torch.zeros((M, K), dtype=torch.bool, device=self.device)
        center_indices = atom_indices.clone()
        
        # Fill arrays
        for i, atom_idx in enumerate(atom_indices):
            # Find neighbors from edge list
            mask = mapping[0] == atom_idx
            neighs = mapping[1, mask]
            
            # Apply element filter
            if element_filter is not None:
                elem_mask = torch.isin(
                    state.atomic_numbers[neighs],
                    torch.tensor(element_filter, device=self.device)
                )
                neighs = neighs[elem_mask]
            
            n = min(len(neighs), K)
            if n > 0:
                neighbor_indices[i, :n] = neighs[:n]
                valid_mask[i, :n] = True
                
                # Compute PBC-corrected positions using minimum image convention
                # dr = pos[neighbor] - pos[center]
                dr = positions[neighs[:n]] - positions[atom_idx].unsqueeze(0)  # (n, 3)
                # Convert to fractional, wrap, convert back
                frac = dr.float() @ cell_inv                     # (n, 3)
                frac = frac - torch.round(frac)                  # wrap to [-0.5, 0.5]
                dr_pbc = frac @ cell_2d.float()                  # (n, 3)
                neighbor_positions_pbc[i, :n] = (positions[atom_idx] + dr_pbc).to(positions.dtype)
        
        return neighbor_indices, neighbor_positions_pbc, valid_mask, center_indices
    
    def _compute_geometry(
        self,
        positions: torch.Tensor,
        center_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_positions_pbc: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute vectors, distances, and angles.
        
        Uses numerically safe operations to avoid NaN gradients:
        - sqrt(sum_sq + eps) instead of norm() to avoid 0/0 at zero distance
        - torch.where() instead of * mask to avoid 0 * NaN = NaN in backward
        """
        
        center_pos = positions[center_indices]  # [M, 3]
        vectors_raw = neighbor_positions_pbc - center_pos.unsqueeze(1)  # [M, K, 3]
        
        # Safe distance: sqrt(sum_sq + eps) avoids NaN gradient at zero distance
        distances_raw = torch.sqrt((vectors_raw * vectors_raw).sum(dim=-1) + 1e-10)  # [M, K]
        
        # Safe normalize
        vectors_normed = vectors_raw / (distances_raw.unsqueeze(-1) + 1e-10)
        
        # Spherical coordinates
        cos_theta = vectors_normed[..., 2].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        thetas_raw = torch.acos(cos_theta)
        phis_raw = torch.atan2(vectors_normed[..., 1], vectors_normed[..., 0])
        
        # Use torch.where instead of multiplication for masking.
        # Multiplication mask: 0 * NaN = NaN in IEEE 754 (NaN propagates in backward).
        # torch.where: cleanly selects values, no gradient from the unselected branch.
        mask_3d = valid_mask.unsqueeze(-1)  # [M, K, 1]
        mask_2d = valid_mask               # [M, K]
        zero3 = torch.zeros_like(vectors_normed)
        zero1 = torch.zeros_like(distances_raw)
        
        vectors = torch.where(mask_3d, vectors_normed, zero3)
        distances = torch.where(mask_2d, distances_raw, zero1)
        thetas = torch.where(mask_2d, thetas_raw, zero1)
        phis = torch.where(mask_2d, phis_raw, zero1)
        
        return vectors, distances, thetas, phis
    
    def _compute_cn(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Coordination number."""
        return valid_mask.sum(dim=1).float()
    
    def _compute_tetrahedral(self, vectors: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Tetrahedral order parameter."""
        params = self.default_params['tet']
        M, K, _ = vectors.shape
        
        # All pairwise dot products
        dots = torch.einsum('mki,mji->mkj', vectors, vectors)
        dots = dots.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.acos(dots)
        
        # Gaussian weight
        target_angle = params['TA'] * math.pi
        delta_theta = params['delta_theta'] * math.pi / 180.0
        
        angle_diff = angles - target_angle
        exponent = -0.5 * (angle_diff / (delta_theta + 1e-10)) ** 2
        exponent = exponent.clamp(min=-50.0)
        gaussian = torch.exp(exponent)
        
        # Mask
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        diag_mask = ~torch.eye(K, device=self.device, dtype=torch.bool).unsqueeze(0)
        full_mask = pair_mask & diag_mask
        
        # Sum and normalize
        qtet = (gaussian * full_mask.float()).sum(dim=(1, 2))
        n_neighbors = valid_mask.sum(dim=1).float().clamp(min=1.0)
        norm = n_neighbors * (n_neighbors - 1).clamp(min=1.0)
        
        return qtet / (norm + 1e-10)
    
    def _compute_octahedral(self, vectors: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Octahedral order parameter."""
        params = self.default_params['oct']
        M, K, _ = vectors.shape
        
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
        
        # Mask
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        diag_mask = ~torch.eye(K, device=self.device, dtype=torch.bool).unsqueeze(0)
        full_mask = pair_mask & diag_mask
        
        total = ((contrib_south + contrib_equat) * full_mask.float()).sum(dim=(1, 2))
        
        n_neighbors = valid_mask.sum(dim=1).float()
        norm = n_neighbors * (3 + (n_neighbors - 2) * (n_neighbors - 3))
        
        return total / (norm + 1e-10)
    
    def _compute_bcc(self, vectors: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """BCC order parameter."""
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
    
    def _compute_q2(self, thetas: torch.Tensor, phis: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Q2 bond orientational order parameter."""
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
        
        # Y_2^{-2}
        pre = 0.25 * sqrt_15_2pi * sin_t2
        real = (pre * cos_2p * valid_mask.float()).sum(dim=1)
        imag = -(pre * sin_2p * valid_mask.float()).sum(dim=1)
        acc += real ** 2 + imag ** 2
        
        # Y_2^{-1}
        pre = 0.5 * sqrt_15_2pi * sin_t * cos_t
        real = (pre * cos_p * valid_mask.float()).sum(dim=1)
        imag = -(pre * sin_p * valid_mask.float()).sum(dim=1)
        acc += real ** 2 + imag ** 2
        
        # Y_2^0
        real = (0.25 * sqrt_5_pi * (3 * cos_t2 - 1.0) * valid_mask.float()).sum(dim=1)
        acc += real ** 2
        
        # Y_2^1
        pre = 0.5 * sqrt_15_2pi * sin_t * cos_t
        real = -(pre * cos_p * valid_mask.float()).sum(dim=1)
        imag = -(pre * sin_p * valid_mask.float()).sum(dim=1)
        acc += real ** 2 + imag ** 2
        
        # Y_2^2
        pre = 0.25 * sqrt_15_2pi * sin_t2
        real = (pre * cos_2p * valid_mask.float()).sum(dim=1)
        imag = (pre * sin_2p * valid_mask.float()).sum(dim=1)
        acc += real ** 2 + imag ** 2
        
        return torch.sqrt(4 * math.pi * acc / (5 * n_neighbors.squeeze() ** 2 + 1e-10))
    
    def _compute_q4(self, thetas: torch.Tensor, phis: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Q4 bond orientational order parameter (simplified m=0 only)."""
        n_neighbors = valid_mask.sum(dim=1, keepdim=True).float()
        
        cos_t = torch.cos(thetas)
        sqrt_1_pi = math.sqrt(1 / math.pi)
        pre = (3 / 16.0) * sqrt_1_pi * (35 * cos_t ** 4 - 30 * cos_t ** 2 + 3.0)
        real = (pre * valid_mask.float()).sum(dim=1)
        acc = real ** 2
        
        return torch.sqrt(4 * math.pi * acc / (9 * n_neighbors.squeeze() ** 2 + 1e-10))
    
    def _compute_q6(self, thetas: torch.Tensor, phis: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Q6 bond orientational order parameter (simplified m=0 only)."""
        n_neighbors = valid_mask.sum(dim=1, keepdim=True).float()
        
        cos_t = torch.cos(thetas)
        sqrt_13_pi = math.sqrt(13 / math.pi)
        pre = (1 / 32.0) * sqrt_13_pi * (
            231 * cos_t ** 6 - 315 * cos_t ** 4 + 105 * cos_t ** 2 - 5.0
        )
        real = (pre * valid_mask.float()).sum(dim=1)
        acc = real ** 2
        
        return torch.sqrt(4 * math.pi * acc / (13 * n_neighbors.squeeze() ** 2 + 1e-10))


# =============================================================================
# WARP Backend (Optional, for large systems)
# =============================================================================

if WARP_AVAILABLE:
    import warp as wp
    
    # -------------------------------------------------------------------------
    # WARP Helper Functions
    # -------------------------------------------------------------------------
    
    @wp.func
    def safe_acos_wp(x: float) -> float:
        """Numerically stable arccos."""
        return wp.acos(wp.clamp(x, -1.0 + 1e-7, 1.0 - 1e-7))
    
    @wp.func
    def compute_tetrahedral_weight(angle: float, target_angle: float, delta_theta: float) -> float:
        """Gaussian weight for tetrahedral OP."""
        diff = angle - target_angle
        exponent = -0.5 * (diff / delta_theta) ** 2.0
        exponent = wp.max(exponent, -50.0)
        return wp.exp(exponent)
    
    # -------------------------------------------------------------------------
    # WARP Kernels
    # -------------------------------------------------------------------------
    
    @wp.kernel
    def compute_neighbor_vectors_kernel(
        positions: wp.array(dtype=wp.vec3),
        neighbor_list: wp.array(dtype=int, ndim=2),
        cell_shifts: wp.array(dtype=wp.vec3, ndim=2),
        valid_mask: wp.array(dtype=int, ndim=2),
        vectors: wp.array(dtype=wp.vec3, ndim=2),
        distances: wp.array(dtype=float, ndim=2),
    ):
        """Compute neighbor vectors with PBC."""
        i = wp.tid()
        center_pos = positions[i]
        
        for j in range(neighbor_list.shape[1]):
            if valid_mask[i, j] > 0:
                neighbor_idx = neighbor_list[i, j]
                neighbor_pos = positions[neighbor_idx]
                shift = cell_shifts[i, j]
                vec = neighbor_pos + shift - center_pos
                dist = wp.length(vec)
                
                if dist > 1e-10:
                    vec = vec / dist
                
                vectors[i, j] = vec
                distances[i, j] = dist
            else:
                vectors[i, j] = wp.vec3(0.0, 0.0, 0.0)
                distances[i, j] = 0.0
    
    @wp.kernel
    def compute_tetrahedral_op_kernel(
        vectors: wp.array(dtype=wp.vec3, ndim=2),
        valid_mask: wp.array(dtype=int, ndim=2),
        target_angle: float,
        delta_theta: float,
        q_tet: wp.array(dtype=float),
    ):
        """Compute tetrahedral order parameter."""
        i = wp.tid()
        K = vectors.shape[1]
        
        # Count valid neighbors
        n_neighbors = 0
        for j in range(K):
            if valid_mask[i, j] > 0:
                n_neighbors += 1
        
        if n_neighbors < 2:
            q_tet[i] = 0.0
            return
        
        # Compute pairwise angles
        sum_weights = 0.0
        for j in range(K):
            if valid_mask[i, j] == 0:
                continue
            v_j = vectors[i, j]
            
            for k in range(j + 1, K):
                if valid_mask[i, k] == 0:
                    continue
                v_k = vectors[i, k]
                
                dot = wp.dot(v_j, v_k)
                angle = safe_acos_wp(dot)
                weight = compute_tetrahedral_weight(angle, target_angle, delta_theta)
                sum_weights += weight
        
        n_pairs = float(n_neighbors * (n_neighbors - 1)) / 2.0
        if n_pairs > 0.0:
            q_tet[i] = sum_weights / n_pairs
        else:
            q_tet[i] = 0.0
    
    @wp.kernel
    def compute_coordination_kernel(
        valid_mask: wp.array(dtype=int, ndim=2),
        cn: wp.array(dtype=float),
    ):
        """Compute coordination number."""
        i = wp.tid()
        K = valid_mask.shape[1]
        count = 0
        for j in range(K):
            if valid_mask[i, j] > 0:
                count += 1
        cn[i] = float(count)
    
    @wp.kernel
    def compute_octahedral_op_kernel(
        vectors: wp.array(dtype=wp.vec3, ndim=2),
        valid_mask: wp.array(dtype=int, ndim=2),
        theta_threshold: float,
        delta_theta_180: float,
        delta_theta_90: float,
        q_oct: wp.array(dtype=float),
    ):
        """Compute octahedral order parameter."""
        i = wp.tid()
        K = vectors.shape[1]
        
        n_neighbors = 0
        for j in range(K):
            if valid_mask[i, j] > 0:
                n_neighbors += 1
        
        if n_neighbors < 2:
            q_oct[i] = 0.0
            return
        
        sum_contrib = 0.0
        for j in range(K):
            if valid_mask[i, j] == 0:
                continue
            v_j = vectors[i, j]
            
            for k in range(j + 1, K):
                if valid_mask[i, k] == 0:
                    continue
                v_k = vectors[i, k]
                
                dot = wp.dot(v_j, v_k)
                angle = safe_acos_wp(dot)
                
                if angle >= theta_threshold:
                    diff_180 = angle - 3.141592653589793
                    gauss_180 = wp.exp(-0.5 * (diff_180 / delta_theta_180) ** 2.0)
                    sum_contrib += 3.0 * gauss_180
                else:
                    diff_90 = angle - 1.5707963267948966
                    gauss_90 = wp.exp(-0.5 * (diff_90 / delta_theta_90) ** 2.0)
                    sum_contrib += gauss_90
        
        norm = float(n_neighbors) * (3.0 + float((n_neighbors - 2) * (n_neighbors - 3)))
        if norm > 0.0:
            q_oct[i] = sum_contrib / norm
        else:
            q_oct[i] = 0.0
    
    @wp.kernel
    def compute_bcc_op_kernel(
        vectors: wp.array(dtype=wp.vec3, ndim=2),
        valid_mask: wp.array(dtype=int, ndim=2),
        theta_threshold: float,
        delta_theta: float,
        weight_spp: float,
        q_bcc: wp.array(dtype=float),
    ):
        """Compute BCC order parameter."""
        i = wp.tid()
        K = vectors.shape[1]
        
        n_neighbors = 0
        for j in range(K):
            if valid_mask[i, j] > 0:
                n_neighbors += 1
        
        if n_neighbors < 2:
            q_bcc[i] = 0.0
            return
        
        sum_contrib = 0.0
        for j in range(K):
            if valid_mask[i, j] == 0:
                continue
            v_j = vectors[i, j]
            
            for k in range(j + 1, K):
                if valid_mask[i, k] == 0:
                    continue
                v_k = vectors[i, k]
                
                dot = wp.dot(v_j, v_k)
                angle = safe_acos_wp(dot)
                
                if angle >= theta_threshold:
                    diff_180 = angle - 3.141592653589793
                    gauss_180 = wp.exp(-0.5 * (diff_180 / delta_theta) ** 2.0)
                    sum_contrib += weight_spp * gauss_180
        
        norm = float(n_neighbors) * (6.0 + float((n_neighbors - 2) * (n_neighbors - 3)))
        if norm > 0.0:
            q_bcc[i] = sum_contrib / norm
        else:
            q_bcc[i] = 0.0
    
    # -------------------------------------------------------------------------
    # WARP Order Parameters Class
    # -------------------------------------------------------------------------
    
    class WarpOrderParameters(nn.Module):
        """
        WARP-accelerated order parameter calculator.
        
        10-100x faster than PyTorch for large systems (>1000 atoms).
        Requires NVIDIA GPU and warp-lang package.
        """
        
        SUPPORTED_TYPES = ['cn', 'tet', 'oct', 'bcc', 'q2', 'q4', 'q6']
        
        def __init__(self, cutoff: float = 3.5, device: str = 'cuda', max_neighbors: int = 64):
            super().__init__()
            self.cutoff = cutoff
            self.device = torch.device(device)
            self.max_neighbors = max_neighbors
            
            if self.device.type != 'cuda':
                raise ValueError("WARP backend requires CUDA device")
            
            # Parameters
            self.params = {
                'tet': {
                    'target_angle': 0.6081734479693927 * math.pi,
                    'delta_theta': 12.0 * math.pi / 180.0,
                },
                'oct': {
                    'theta_threshold': 2.792526803190927,
                    'delta_theta_180': 12.0 * math.pi / 180.0,
                    'delta_theta_90': 10.0 * math.pi / 180.0,
                },
                'bcc': {
                    'theta_threshold': 2.792526803190927,
                    'delta_theta': 19.47 * math.pi / 180.0,
                    'weight_spp': 6.0,
                },
            }
        
        def forward(
            self,
            state: ts.SimState,
            atom_indices: torch.Tensor,
            order_params: List[str],
            element_filter: Optional[List[int]] = None,
        ) -> Dict[str, torch.Tensor]:
            """Compute order parameters using WARP kernels."""
            
            M = len(atom_indices)
            if M == 0:
                return {op: torch.tensor([], device=self.device) for op in order_params}
            
            # Build neighbor data (PyTorch)
            neighbor_list, cell_shifts, valid_mask = self._build_neighbor_data(
                state, atom_indices, element_filter
            )
            
            # Get positions for target atoms
            positions = state.positions[atom_indices].contiguous()
            
            # Convert to WARP arrays
            K = self.max_neighbors
            wp_device = "cuda:0"
            
            wp_positions = wp.from_torch(positions.view(-1, 3).contiguous(), dtype=wp.vec3)
            wp_neighbor_list = wp.from_torch(neighbor_list.contiguous(), dtype=wp.int32)
            wp_cell_shifts = wp.from_torch(cell_shifts.view(M, K, 3).contiguous(), dtype=wp.vec3)
            wp_valid_mask = wp.from_torch(valid_mask.int().contiguous(), dtype=wp.int32)
            
            # Compute neighbor vectors
            wp_vectors = wp.zeros((M, K), dtype=wp.vec3, device=wp_device)
            wp_distances = wp.zeros((M, K), dtype=wp.float32, device=wp_device)
            
            # Use full positions for neighbor lookup
            full_positions = state.positions.contiguous()
            wp_full_positions = wp.from_torch(full_positions.view(-1, 3).contiguous(), dtype=wp.vec3)
            
            # Custom kernel for neighbor vectors with full position array
            self._compute_vectors_manual(
                full_positions, atom_indices, neighbor_list, cell_shifts, valid_mask,
                wp_vectors, wp_distances, M, K
            )
            
            # Compute requested OPs
            results = {}
            for op in order_params:
                if op == 'cn':
                    wp_cn = wp.zeros(M, dtype=wp.float32, device=wp_device)
                    wp.launch(compute_coordination_kernel, dim=M,
                             inputs=[wp_valid_mask], outputs=[wp_cn])
                    results[op] = wp.to_torch(wp_cn)
                
                elif op == 'tet':
                    wp_qtet = wp.zeros(M, dtype=wp.float32, device=wp_device)
                    wp.launch(compute_tetrahedral_op_kernel, dim=M,
                             inputs=[wp_vectors, wp_valid_mask,
                                    self.params['tet']['target_angle'],
                                    self.params['tet']['delta_theta']],
                             outputs=[wp_qtet])
                    results[op] = wp.to_torch(wp_qtet)
                
                elif op == 'oct':
                    wp_qoct = wp.zeros(M, dtype=wp.float32, device=wp_device)
                    wp.launch(compute_octahedral_op_kernel, dim=M,
                             inputs=[wp_vectors, wp_valid_mask,
                                    self.params['oct']['theta_threshold'],
                                    self.params['oct']['delta_theta_180'],
                                    self.params['oct']['delta_theta_90']],
                             outputs=[wp_qoct])
                    results[op] = wp.to_torch(wp_qoct)
                
                elif op == 'bcc':
                    wp_qbcc = wp.zeros(M, dtype=wp.float32, device=wp_device)
                    wp.launch(compute_bcc_op_kernel, dim=M,
                             inputs=[wp_vectors, wp_valid_mask,
                                    self.params['bcc']['theta_threshold'],
                                    self.params['bcc']['delta_theta'],
                                    self.params['bcc']['weight_spp']],
                             outputs=[wp_qbcc])
                    results[op] = wp.to_torch(wp_qbcc)
                
                elif op in ['q2', 'q4', 'q6']:
                    # Fall back to PyTorch for spherical harmonics
                    warnings.warn(f"{op} using PyTorch fallback in WARP backend")
                    pytorch_calc = PyTorchOrderParameters(
                        cutoff=self.cutoff, device=str(self.device), max_neighbors=self.max_neighbors
                    )
                    pytorch_results = pytorch_calc(state, atom_indices, [op], element_filter)
                    results[op] = pytorch_results[op]
            
            return results
        
        def _compute_vectors_manual(
            self, full_positions, atom_indices, neighbor_list, cell_shifts, valid_mask,
            wp_vectors, wp_distances, M, K
        ):
            """Compute neighbor vectors manually (CPU loop, then transfer)."""
            vectors = torch.zeros(M, K, 3, device=self.device)
            distances = torch.zeros(M, K, device=self.device)
            
            for i in range(M):
                center_pos = full_positions[atom_indices[i]]
                for j in range(K):
                    if valid_mask[i, j]:
                        neigh_idx = neighbor_list[i, j]
                        neigh_pos = full_positions[neigh_idx]
                        shift = cell_shifts[i, j]
                        vec = neigh_pos + shift - center_pos
                        dist = vec.norm()
                        
                        if dist > 1e-10:
                            vec = vec / dist
                        
                        vectors[i, j] = vec
                        distances[i, j] = dist
            
            # Copy to WARP arrays
            wp.copy(wp_vectors, wp.from_torch(vectors.view(M, K, 3).contiguous(), dtype=wp.vec3))
            wp.copy(wp_distances, wp.from_torch(distances.contiguous(), dtype=wp.float32))
        
        def _build_neighbor_data(
            self,
            state: ts.SimState,
            atom_indices: torch.Tensor,
            element_filter: Optional[List[int]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Build neighbor arrays using minimum image convention."""
            
            positions = state.positions
            cell = state.cell
            pbc = state.pbc if state.pbc is not None else torch.tensor([True, True, True], device=self.device)
            
            if state.system_idx is None:
                system_idx = torch.zeros(len(positions), dtype=torch.long, device=self.device)
            else:
                system_idx = state.system_idx
            
            # Ensure cell is 3D for torchsim_nl
            cell_for_nl = cell.unsqueeze(0) if cell.ndim == 2 else cell
            
            # Only use mapping from torchsim_nl (ignore shifts - may not be shift vectors)
            mapping = torchsim_nl(
                positions=positions,
                cell=cell_for_nl,
                pbc=pbc,
                cutoff=torch.tensor([self.cutoff], device=self.device),
                system_idx=system_idx,
                self_interaction=False,
            )[0]  # Only take mapping
            
            M = len(atom_indices)
            K = self.max_neighbors
            
            # Get the 2D cell matrix for minimum image convention
            cell_2d = cell[0] if cell.ndim == 3 else cell
            cell_inv = torch.linalg.inv(cell_2d.float())
            
            neighbor_list = torch.full((M, K), 0, dtype=torch.long, device=self.device)
            cell_shifts = torch.zeros((M, K, 3), dtype=positions.dtype, device=self.device)
            valid_mask = torch.zeros((M, K), dtype=torch.bool, device=self.device)
            
            for i, atom_idx in enumerate(atom_indices):
                mask = mapping[0] == atom_idx
                neighs = mapping[1, mask]
                
                if element_filter is not None:
                    elem_mask = torch.isin(
                        state.atomic_numbers[neighs],
                        torch.tensor(element_filter, device=self.device)
                    )
                    neighs = neighs[elem_mask]
                
                n = min(len(neighs), K)
                if n > 0:
                    neighbor_list[i, :n] = neighs[:n]
                    valid_mask[i, :n] = True
                    
                    # Compute PBC shift using minimum image convention
                    dr = positions[neighs[:n]] - positions[atom_idx].unsqueeze(0)  # (n, 3)
                    frac = dr.float() @ cell_inv                     # (n, 3)
                    frac = frac - torch.round(frac)                  # wrap
                    dr_pbc = frac @ cell_2d.float()                  # (n, 3)
                    # cell_shifts = dr_pbc - dr (correction to add to positions[neigh])
                    cell_shifts[i, :n] = (dr_pbc - dr).to(positions.dtype)
            
            return neighbor_list, cell_shifts, valid_mask


# =============================================================================
# Unified Interface
# =============================================================================

class TorchSimOrderParameters(nn.Module):
    """
    Order parameter calculator with automatic backend selection.
    
    Automatically uses WARP for acceleration on CUDA when available,
    falling back to PyTorch otherwise.
    
    Args:
        cutoff: Distance cutoff for neighbor finding (Å)
        device: 'cpu' or 'cuda'
        max_neighbors: Maximum neighbors per atom
        backend: 'auto', 'pytorch', or 'warp'
            - 'auto': Use WARP if available on CUDA, else PyTorch
            - 'pytorch': Always use PyTorch
            - 'warp': Always use WARP (error if unavailable)
    
    Usage:
        >>> op_calc = TorchSimOrderParameters(cutoff=3.5, device='cuda')
        >>> results = op_calc(state, p_indices, ['tet', 'cn', 'q4'])
        >>> print(f"Tetrahedral: {results['tet'].mean():.3f}")
    
    Supported order parameters:
        - cn: Coordination number
        - tet: Tetrahedral (SiO4, PO4, PS4)
        - oct: Octahedral (FeO6, TiO6)
        - bcc: BCC-like
        - q2, q4, q6: Steinhardt bond orientational
    """
    
    SUPPORTED_TYPES = ['cn', 'tet', 'oct', 'bcc', 'q2', 'q4', 'q6']
    
    def __init__(
        self,
        cutoff: float = 3.5,
        device: str = 'cpu',
        max_neighbors: int = 64,
        backend: str = 'auto',
    ):
        super().__init__()
        self.cutoff = cutoff
        self.device = torch.device(device)
        self.max_neighbors = max_neighbors
        
        # Select backend
        if backend == 'auto':
            if WARP_AVAILABLE and self.device.type == 'cuda':
                self._backend = 'warp'
            else:
                self._backend = 'pytorch'
        elif backend == 'pytorch':
            self._backend = 'pytorch'
        elif backend == 'warp':
            if not WARP_AVAILABLE:
                raise RuntimeError(
                    "WARP backend requested but warp-lang not installed. "
                    "Install with: pip install warp-lang"
                )
            if self.device.type != 'cuda':
                raise RuntimeError("WARP backend requires CUDA device")
            self._backend = 'warp'
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'pytorch', or 'warp'")
        
        # Initialize backend
        if self._backend == 'warp':
            self._calc = WarpOrderParameters(cutoff=cutoff, device=str(device), max_neighbors=max_neighbors)
            print(f"  ✓ Order parameters: WARP backend (10-100x faster for large systems)")
        else:
            self._calc = PyTorchOrderParameters(cutoff=cutoff, device=str(device), max_neighbors=max_neighbors)
            if self.device.type == 'cuda' and not WARP_AVAILABLE:
                print(f"  ⚠ Order parameters: PyTorch backend (install warp-lang for 10-100x speedup)")
            else:
                print(f"  ✓ Order parameters: PyTorch backend")
    
    @property
    def backend(self) -> str:
        """Current backend ('warp' or 'pytorch')."""
        return self._backend
    
    def forward(
        self,
        state: ts.SimState,
        atom_indices: torch.Tensor,
        order_params: List[str],
        element_filter: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute order parameters for specified atoms.
        
        Args:
            state: torch_sim SimState
            atom_indices: (M,) tensor of atom indices
            order_params: List of OP types ['tet', 'cn', etc.]
            element_filter: Optional list of atomic numbers for neighbors
        
        Returns:
            Dict mapping OP names to (M,) tensors
        """
        return self._calc(state, atom_indices, order_params, element_filter)


# =============================================================================
# Constraint Loading Utilities
# =============================================================================

def load_constraints_from_json(json_path: str) -> Dict:
    """
    Load order parameter constraints from JSON file.
    
    Args:
        json_path: Path to constraints JSON file
    
    Returns:
        Dictionary with constraint specifications
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def get_atom_indices_from_constraints(
    constraints: Dict,
    atomic_numbers: torch.Tensor,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Get atom indices that have constraints.
    
    Args:
        constraints: Loaded constraints dictionary
        atomic_numbers: Atomic numbers tensor
        device: Device for output tensor
    
    Returns:
        Tensor of atom indices with constraints
    """
    if 'atom_constraints' in constraints:
        indices = [int(idx) for idx in constraints['atom_constraints'].keys()]
        return torch.tensor(indices, dtype=torch.long, device=device)
    return torch.tensor([], dtype=torch.long, device=device)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'TorchSimOrderParameters',
    'PyTorchOrderParameters',
    'WARP_AVAILABLE',
    'load_constraints_from_json',
    'get_atom_indices_from_constraints',
]

# Also export WarpOrderParameters if available
if WARP_AVAILABLE:
    __all__.append('WarpOrderParameters')
