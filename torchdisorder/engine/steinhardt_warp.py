"""
Warp-Accelerated Full Steinhardt Bond-Orientational Order Parameters
=====================================================================

Computes q_l per atom using the FULL sum over all m from -l to +l:

    q_l_i = sqrt(4π/(2l+1) * sum_m |q_lm_i|²)
    q_lm_i = (1/N_b) * sum_j Y_lm(r̂_ij)

where Y_lm are real spherical harmonics:

    Y_l0^R  = N_l0 * P_l^0(z)
    Y_lm^R  = N_lm * lpmv(m, l, z) * cos(m*φ)   (m > 0)
    Y_l-m^R = N_lm * lpmv(m, l, z) * sin(m*φ)   (m > 0)

and N_lm = sqrt(2) * sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)  for m > 0
         = sqrt((2l+1)/(4π))                               for m = 0

The associated Legendre polynomials lpmv(m, l, z) include the
Condon-Shortley phase (-1)^m (matching scipy.special.lpmv convention).

Hardcoded N_lm values (precomputed from the formula above):
  l=4: N_40=0.846284375321634, N_41=0.267618617422916,
       N_42=0.063078313050504, N_43=0.016858388283618, N_44=0.005960340337611
  l=6: N_60=1.017107236282055, N_61=0.221950995245231,
       N_62=0.035093533695807, N_63=0.005848922282634, N_64=0.001067862223764,
       N_65=0.000227668991076,  N_66=0.000065722376642

Hardcoded lpmv formulas for l=4, l=6 in terms of z=cos(θ), s=sin(θ):
  lpmv(0,4,z) = (35z⁴ - 30z² + 3)/8
  lpmv(1,4,z) = -5z*s*(7z²-3)/2
  lpmv(2,4,z) = 15s²(7z²-1)/2
  lpmv(3,4,z) = -105z*s³
  lpmv(4,4,z) = 105s⁴

  lpmv(0,6,z) = (231z⁶ - 315z⁴ + 105z² - 5)/16
  lpmv(1,6,z) = -21z*s*(33z⁴ - 30z² + 5)/8
  lpmv(2,6,z) = 105s²*(33z⁴ - 18z² + 1)/8
  lpmv(3,6,z) = -315z*s³*(11z² - 3)/2
  lpmv(4,6,z) = 945s⁴*(11z² - 1)/2
  lpmv(5,6,z) = -10395z*s⁵
  lpmv(6,6,z) = 10395s⁶

All formulas verified against scipy.special.lpmv at multiple test points.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# WARP availability check (same pattern as order_params.py)
# ---------------------------------------------------------------------------

WARP_AVAILABLE = False
try:
    import warp as wp
    wp.init()
    # Warp kernels require a CUDA device; CPU-only builds cannot run them
    if wp.is_cuda_available():
        WARP_AVAILABLE = True
except (ImportError, Exception):
    pass

# ---------------------------------------------------------------------------
# WARP kernels (only defined when warp is available)
# ---------------------------------------------------------------------------

if WARP_AVAILABLE:
    import warp as wp

    # -----------------------------------------------------------------------
    # Real Spherical Harmonic helper functions
    # -----------------------------------------------------------------------

    @wp.func
    def _plm4(m_idx: int, z: float, s: float) -> float:
        """Return lpmv(m, 4, z) for m = m_idx in [0..4].

        lpmv includes the Condon-Shortley phase (-1)^m, matching scipy.
        """
        if m_idx == 0:
            return (35.0*z*z*z*z - 30.0*z*z + 3.0) / 8.0
        elif m_idx == 1:
            return -5.0 * z * s * (7.0*z*z - 3.0) / 2.0
        elif m_idx == 2:
            return 15.0 * s*s * (7.0*z*z - 1.0) / 2.0
        elif m_idx == 3:
            return -105.0 * z * s*s*s
        else:  # m_idx == 4
            return 105.0 * s*s*s*s

    @wp.func
    def _plm6(m_idx: int, z: float, s: float) -> float:
        """Return lpmv(m, 6, z) for m = m_idx in [0..6].

        lpmv includes the Condon-Shortley phase (-1)^m, matching scipy.
        """
        if m_idx == 0:
            return (231.0*z*z*z*z*z*z - 315.0*z*z*z*z + 105.0*z*z - 5.0) / 16.0
        elif m_idx == 1:
            return -21.0 * z * s * (33.0*z*z*z*z - 30.0*z*z + 5.0) / 8.0
        elif m_idx == 2:
            return 105.0 * s*s * (33.0*z*z*z*z - 18.0*z*z + 1.0) / 8.0
        elif m_idx == 3:
            return -315.0 * z * s*s*s * (11.0*z*z - 3.0) / 2.0
        elif m_idx == 4:
            return 945.0 * s*s*s*s * (11.0*z*z - 1.0) / 2.0
        elif m_idx == 5:
            return -10395.0 * z * s*s*s*s*s
        else:  # m_idx == 6
            return 10395.0 * s*s*s*s*s*s

    @wp.func
    def qlm_real_l4(m_idx: int, z: float, s: float, cp: float, sp: float) -> float:
        """Return Y_4m^real for m_idx = 0..8, where m = m_idx - 4 (so m in [-4..4]).

        Y_l0^R  = N_l0 * P_l^0(z)
        Y_lm^R  = N_lm * lpmv(m,l,z) * cos(m*phi)   (m > 0)
        Y_l-m^R = N_lm * lpmv(m,l,z) * sin(m*phi)   (m > 0, here |m|)

        Normalization (real SH, orthonormal):
          N_40 = 0.846284375321634
          N_41 = 0.267618617422916
          N_42 = 0.063078313050504
          N_43 = 0.016858388283618
          N_44 = 0.005960340337611

        cos/sin multiples via Chebyshev recurrence (2cp*c_{k-1} - c_{k-2}).
        """
        # m_idx: 0=m-4, 1=m-3, 2=m-2, 3=m-1, 4=m0, 5=m+1, 6=m+2, 7=m+3, 8=m+4
        # Chebyshev recurrence for cos(k*phi) and sin(k*phi)
        c1 = cp
        s1 = sp
        c2 = 2.0*cp*c1 - 1.0          # cos(2phi)
        s2 = 2.0*cp*s1 - 0.0          # sin(2phi)
        c3 = 2.0*cp*c2 - c1           # cos(3phi)
        s3 = 2.0*cp*s2 - s1           # sin(3phi)
        c4 = 2.0*cp*c3 - c2           # cos(4phi)
        s4 = 2.0*cp*s3 - s2           # sin(4phi)

        if m_idx == 4:  # m = 0
            return float(0.846284375321634) * _plm4(0, z, s)
        elif m_idx == 5:  # m = +1: N_41 * P_41(z) * cos(phi)
            return float(0.267618617422916) * _plm4(1, z, s) * c1
        elif m_idx == 3:  # m = -1: N_41 * P_41(z) * sin(phi)
            return float(0.267618617422916) * _plm4(1, z, s) * s1
        elif m_idx == 6:  # m = +2: N_42 * P_42(z) * cos(2phi)
            return float(0.063078313050504) * _plm4(2, z, s) * c2
        elif m_idx == 2:  # m = -2: N_42 * P_42(z) * sin(2phi)
            return float(0.063078313050504) * _plm4(2, z, s) * s2
        elif m_idx == 7:  # m = +3: N_43 * P_43(z) * cos(3phi)
            return float(0.016858388283618) * _plm4(3, z, s) * c3
        elif m_idx == 1:  # m = -3: N_43 * P_43(z) * sin(3phi)
            return float(0.016858388283618) * _plm4(3, z, s) * s3
        elif m_idx == 8:  # m = +4: N_44 * P_44(z) * cos(4phi)
            return float(0.005960340337611) * _plm4(4, z, s) * c4
        else:  # m_idx == 0, m = -4: N_44 * P_44(z) * sin(4phi)
            return float(0.005960340337611) * _plm4(4, z, s) * s4

    @wp.func
    def qlm_real_l6(m_idx: int, z: float, s: float, cp: float, sp: float) -> float:
        """Return Y_6m^real for m_idx = 0..12, where m = m_idx - 6 (so m in [-6..6]).

        Normalization:
          N_60 = 1.017107236282055
          N_61 = 0.221950995245231
          N_62 = 0.035093533695807
          N_63 = 0.005848922282634
          N_64 = 0.001067862223764
          N_65 = 0.000227668991076
          N_66 = 0.000065722376642
        """
        # Chebyshev recurrence for cos/sin multiples
        c1 = cp
        s1 = sp
        c2 = 2.0*cp*c1 - 1.0
        s2 = 2.0*cp*s1 - 0.0
        c3 = 2.0*cp*c2 - c1
        s3 = 2.0*cp*s2 - s1
        c4 = 2.0*cp*c3 - c2
        s4 = 2.0*cp*s3 - s2
        c5 = 2.0*cp*c4 - c3
        s5 = 2.0*cp*s4 - s3
        c6 = 2.0*cp*c5 - c4
        s6 = 2.0*cp*s5 - s4

        if m_idx == 6:  # m = 0
            return float(1.017107236282055) * _plm6(0, z, s)
        elif m_idx == 7:  # m = +1
            return float(0.221950995245231) * _plm6(1, z, s) * c1
        elif m_idx == 5:  # m = -1
            return float(0.221950995245231) * _plm6(1, z, s) * s1
        elif m_idx == 8:  # m = +2
            return float(0.035093533695807) * _plm6(2, z, s) * c2
        elif m_idx == 4:  # m = -2
            return float(0.035093533695807) * _plm6(2, z, s) * s2
        elif m_idx == 9:  # m = +3
            return float(0.005848922282634) * _plm6(3, z, s) * c3
        elif m_idx == 3:  # m = -3
            return float(0.005848922282634) * _plm6(3, z, s) * s3
        elif m_idx == 10:  # m = +4
            return float(0.001067862223764) * _plm6(4, z, s) * c4
        elif m_idx == 2:  # m = -4
            return float(0.001067862223764) * _plm6(4, z, s) * s4
        elif m_idx == 11:  # m = +5
            return float(0.000227668991076) * _plm6(5, z, s) * c5
        elif m_idx == 1:  # m = -5
            return float(0.000227668991076) * _plm6(5, z, s) * s5
        elif m_idx == 12:  # m = +6
            return float(0.000065722376642) * _plm6(6, z, s) * c6
        else:  # m_idx == 0, m = -6
            return float(0.000065722376642) * _plm6(6, z, s) * s6

    # -----------------------------------------------------------------------
    # Steinhardt q4 kernel
    # -----------------------------------------------------------------------

    @wp.kernel
    def steinhardt_q4_kernel(
        vectors: wp.array(dtype=wp.vec3, ndim=2),    # (N_atoms, max_neighbors) unit vecs
        valid_mask: wp.array(dtype=int, ndim=2),      # (N_atoms, max_neighbors)
        ql_out: wp.array(dtype=float),                # (N_atoms,) output
    ):
        """Full Steinhardt q4 using real SH, summing m = -4..+4."""
        i = wp.tid()
        K = vectors.shape[1]

        # Accumulate q_lm for each of the 9 m values (m = -4 .. +4, i.e. m_idx 0..8)
        acc0 = float(0.0)
        acc1 = float(0.0)
        acc2 = float(0.0)
        acc3 = float(0.0)
        acc4 = float(0.0)
        acc5 = float(0.0)
        acc6 = float(0.0)
        acc7 = float(0.0)
        acc8 = float(0.0)
        n_b = int(0)

        for j in range(K):
            if valid_mask[i, j] == 0:
                continue
            u = vectors[i, j]
            x = u[0]
            y = u[1]
            z = u[2]

            s_sq = x*x + y*y
            s = wp.sqrt(s_sq)

            inv_s = float(0.0)
            if s > float(1e-10):
                inv_s = float(1.0) / s

            cp = x * inv_s
            sp = y * inv_s

            acc0 += qlm_real_l4(0, z, s, cp, sp)
            acc1 += qlm_real_l4(1, z, s, cp, sp)
            acc2 += qlm_real_l4(2, z, s, cp, sp)
            acc3 += qlm_real_l4(3, z, s, cp, sp)
            acc4 += qlm_real_l4(4, z, s, cp, sp)
            acc5 += qlm_real_l4(5, z, s, cp, sp)
            acc6 += qlm_real_l4(6, z, s, cp, sp)
            acc7 += qlm_real_l4(7, z, s, cp, sp)
            acc8 += qlm_real_l4(8, z, s, cp, sp)
            n_b += 1

        if n_b == 0:
            ql_out[i] = float(0.0)
            return

        inv_nb = float(1.0) / float(n_b)
        q0 = acc0 * inv_nb
        q1 = acc1 * inv_nb
        q2 = acc2 * inv_nb
        q3 = acc3 * inv_nb
        q4 = acc4 * inv_nb
        q5 = acc5 * inv_nb
        q6 = acc6 * inv_nb
        q7 = acc7 * inv_nb
        q8 = acc8 * inv_nb

        # q_l = sqrt(4pi/(2l+1) * sum_m qlm^2)
        # For l=4: 4pi/(2*4+1) = 4pi/9
        prefactor = float(4.0 * 3.14159265358979323846) / float(9.0)
        sum_sq = (q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4
                  + q5*q5 + q6*q6 + q7*q7 + q8*q8)
        ql_out[i] = wp.sqrt(prefactor * sum_sq)

    # -----------------------------------------------------------------------
    # Steinhardt q6 kernel
    # -----------------------------------------------------------------------

    @wp.kernel
    def steinhardt_q6_kernel(
        vectors: wp.array(dtype=wp.vec3, ndim=2),    # (N_atoms, max_neighbors) unit vecs
        valid_mask: wp.array(dtype=int, ndim=2),      # (N_atoms, max_neighbors)
        ql_out: wp.array(dtype=float),                # (N_atoms,) output
    ):
        """Full Steinhardt q6 using real SH, summing m = -6..+6."""
        i = wp.tid()
        K = vectors.shape[1]

        # 13 m values: m_idx = 0..12 (m = -6..+6)
        acc0  = float(0.0)
        acc1  = float(0.0)
        acc2  = float(0.0)
        acc3  = float(0.0)
        acc4  = float(0.0)
        acc5  = float(0.0)
        acc6  = float(0.0)
        acc7  = float(0.0)
        acc8  = float(0.0)
        acc9  = float(0.0)
        acc10 = float(0.0)
        acc11 = float(0.0)
        acc12 = float(0.0)
        n_b = int(0)

        for j in range(K):
            if valid_mask[i, j] == 0:
                continue
            u = vectors[i, j]
            x = u[0]
            y = u[1]
            z = u[2]

            s_sq = x*x + y*y
            s = wp.sqrt(s_sq)

            inv_s = float(0.0)
            if s > float(1e-10):
                inv_s = float(1.0) / s

            cp = x * inv_s
            sp = y * inv_s

            acc0  += qlm_real_l6(0,  z, s, cp, sp)
            acc1  += qlm_real_l6(1,  z, s, cp, sp)
            acc2  += qlm_real_l6(2,  z, s, cp, sp)
            acc3  += qlm_real_l6(3,  z, s, cp, sp)
            acc4  += qlm_real_l6(4,  z, s, cp, sp)
            acc5  += qlm_real_l6(5,  z, s, cp, sp)
            acc6  += qlm_real_l6(6,  z, s, cp, sp)
            acc7  += qlm_real_l6(7,  z, s, cp, sp)
            acc8  += qlm_real_l6(8,  z, s, cp, sp)
            acc9  += qlm_real_l6(9,  z, s, cp, sp)
            acc10 += qlm_real_l6(10, z, s, cp, sp)
            acc11 += qlm_real_l6(11, z, s, cp, sp)
            acc12 += qlm_real_l6(12, z, s, cp, sp)
            n_b += 1

        if n_b == 0:
            ql_out[i] = float(0.0)
            return

        inv_nb = float(1.0) / float(n_b)
        q0  = acc0  * inv_nb
        q1  = acc1  * inv_nb
        q2  = acc2  * inv_nb
        q3  = acc3  * inv_nb
        q4  = acc4  * inv_nb
        q5  = acc5  * inv_nb
        q6  = acc6  * inv_nb
        q7  = acc7  * inv_nb
        q8  = acc8  * inv_nb
        q9  = acc9  * inv_nb
        q10 = acc10 * inv_nb
        q11 = acc11 * inv_nb
        q12 = acc12 * inv_nb

        # For l=6: 4pi/(2*6+1) = 4pi/13
        prefactor = float(4.0 * 3.14159265358979323846) / float(13.0)
        sum_sq = (q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6
                  + q7*q7 + q8*q8 + q9*q9 + q10*q10 + q11*q11 + q12*q12)
        ql_out[i] = wp.sqrt(prefactor * sum_sq)


# ---------------------------------------------------------------------------
# Public API: warp-accelerated path
# ---------------------------------------------------------------------------

def steinhardt_ql_warp(
    l: int,
    vectors: np.ndarray,       # (N_atoms, max_neighbors, 3) float32 UNIT vectors
    valid_mask: np.ndarray,    # (N_atoms, max_neighbors) int32
    device: str = "cpu",
) -> np.ndarray:
    """Compute full Steinhardt q_l via warp kernel.

    Parameters
    ----------
    l : int
        Degree of the spherical harmonic. Must be 4 or 6.
    vectors : np.ndarray, shape (N_atoms, max_neighbors, 3), float32
        Pre-normalised unit bond vectors. Invalid slots are ignored via mask.
    valid_mask : np.ndarray, shape (N_atoms, max_neighbors), int32
        1 for a valid neighbor slot, 0 otherwise.
    device : str
        ``"cuda"`` or ``"cpu"``. Currently warp only runs on CUDA, but the
        argument is accepted for API symmetry (falls back to PyTorch if CPU).

    Returns
    -------
    np.ndarray, shape (N_atoms,), float32
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("warp is not available; use steinhardt_ql_pytorch instead")

    if l not in (4, 6):
        raise ValueError(f"steinhardt_ql_warp supports l=4 and l=6 only, got l={l}")

    N_atoms = vectors.shape[0]
    K = vectors.shape[1]

    # Ensure contiguous float32 / int32
    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32).reshape(N_atoms, K, 3)
    mask_i32 = np.ascontiguousarray(valid_mask, dtype=np.int32).reshape(N_atoms, K)

    # Warp device string
    if "cuda" in device.lower() or device.lower() == "gpu":
        wp_device = "cuda:0"
    else:
        # warp kernels require a GPU; fall back to pytorch
        return steinhardt_ql_pytorch(l, vectors, valid_mask)

    # Build warp arrays
    wp_vectors = wp.array(vectors_f32, dtype=wp.vec3, device=wp_device)
    wp_mask = wp.array(mask_i32, dtype=wp.int32, device=wp_device)
    wp_out = wp.zeros(N_atoms, dtype=wp.float32, device=wp_device)

    if l == 4:
        wp.launch(steinhardt_q4_kernel, dim=N_atoms,
                  inputs=[wp_vectors, wp_mask, wp_out], device=wp_device)
    else:  # l == 6
        wp.launch(steinhardt_q6_kernel, dim=N_atoms,
                  inputs=[wp_vectors, wp_mask, wp_out], device=wp_device)

    wp.synchronize()
    return wp.to_torch(wp_out).cpu().numpy()


# ---------------------------------------------------------------------------
# Public API: pure NumPy/PyTorch fallback (same real SH formulas)
# ---------------------------------------------------------------------------

def _lpmv4_np(m: int, z: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Vectorised lpmv(m, 4, z) with Condon-Shortley phase, for m in 0..4."""
    if m == 0:
        return (35.0*z**4 - 30.0*z**2 + 3.0) / 8.0
    elif m == 1:
        return -5.0 * z * s * (7.0*z**2 - 3.0) / 2.0
    elif m == 2:
        return 15.0 * s**2 * (7.0*z**2 - 1.0) / 2.0
    elif m == 3:
        return -105.0 * z * s**3
    else:  # m == 4
        return 105.0 * s**4


def _lpmv6_np(m: int, z: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Vectorised lpmv(m, 6, z) with Condon-Shortley phase, for m in 0..6."""
    if m == 0:
        return (231.0*z**6 - 315.0*z**4 + 105.0*z**2 - 5.0) / 16.0
    elif m == 1:
        return -21.0 * z * s * (33.0*z**4 - 30.0*z**2 + 5.0) / 8.0
    elif m == 2:
        return 105.0 * s**2 * (33.0*z**4 - 18.0*z**2 + 1.0) / 8.0
    elif m == 3:
        return -315.0 * z * s**3 * (11.0*z**2 - 3.0) / 2.0
    elif m == 4:
        return 945.0 * s**4 * (11.0*z**2 - 1.0) / 2.0
    elif m == 5:
        return -10395.0 * z * s**5
    else:  # m == 6
        return 10395.0 * s**6


# Normalization constants (N_lm for real SH)
_N_LM = {
    4: [0.846284375321634, 0.267618617422916, 0.063078313050504,
        0.016858388283618, 0.005960340337611],
    6: [1.017107236282055, 0.221950995245231, 0.035093533695807,
        0.005848922282634, 0.001067862223764, 0.000227668991076,
        0.000065722376642],
}


def steinhardt_ql_pytorch(
    l: int,
    vectors: np.ndarray,    # (N_atoms, max_neighbors, 3) float32 unit vectors
    valid_mask: np.ndarray, # (N_atoms, max_neighbors) int32/bool
) -> np.ndarray:
    """Compute full Steinhardt q_l via vectorised NumPy (no warp needed).

    Uses exactly the same real SH formulas as the warp kernel.
    The ``pytorch`` name is kept for API consistency but the implementation
    is pure NumPy to avoid any torch dependency here.

    Parameters
    ----------
    l : int
        Degree, must be 4 or 6.
    vectors : ndarray (N_atoms, max_neighbors, 3)
        Pre-normalised unit bond vectors.
    valid_mask : ndarray (N_atoms, max_neighbors)
        1 = valid neighbor, 0 = padding.

    Returns
    -------
    ndarray, shape (N_atoms,), float32
    """
    if l not in (4, 6):
        raise ValueError(f"steinhardt_ql_pytorch supports l=4 and l=6 only, got l={l}")

    N_atoms, K, _ = vectors.shape
    mask_f = valid_mask.astype(np.float32)  # (N, K)

    x = vectors[..., 0]  # (N, K)
    y = vectors[..., 1]
    z = vectors[..., 2]

    s_sq = x**2 + y**2
    s = np.sqrt(s_sq)

    # Safe phi components
    safe_s = np.where(s > 1e-10, s, np.ones_like(s))
    cp = np.where(s > 1e-10, x / safe_s, np.zeros_like(x))
    sp = np.where(s > 1e-10, y / safe_s, np.zeros_like(y))

    # Chebyshev recurrence for cos/sin multiples
    m_max = l
    c = [np.ones_like(cp), cp]
    s_trig = [np.zeros_like(sp), sp]
    for k in range(2, m_max + 1):
        c.append(2.0 * cp * c[-1] - c[-2])
        s_trig.append(2.0 * cp * s_trig[-1] - s_trig[-2])

    # Count valid neighbors per atom
    n_b = mask_f.sum(axis=1).clip(min=1e-10)  # (N,)

    N_coeffs = _N_LM[l]
    lpmv_fn = _lpmv4_np if l == 4 else _lpmv6_np
    prefactor = 4.0 * np.pi / (2.0 * l + 1.0)

    sum_sq = np.zeros(N_atoms, dtype=np.float64)

    for m in range(0, l + 1):
        P = lpmv_fn(m, z, s)   # (N, K)
        N_m = N_coeffs[m]

        if m == 0:
            # Y_l0^R = N_m * P
            ylm_r = N_m * P
            # average over neighbors
            qlm = (ylm_r * mask_f).sum(axis=1) / n_b
            sum_sq += qlm ** 2
        else:
            # m > 0: Y_lm^R = N_m * P * cos(m*phi)
            ylm_pos = N_m * P * c[m]
            qlm_pos = (ylm_pos * mask_f).sum(axis=1) / n_b
            sum_sq += qlm_pos ** 2

            # m < 0: Y_l-m^R = N_m * P * sin(m*phi)
            ylm_neg = N_m * P * s_trig[m]
            qlm_neg = (ylm_neg * mask_f).sum(axis=1) / n_b
            sum_sq += qlm_neg ** 2

    return np.sqrt(prefactor * sum_sq).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: build padded (N_atoms, max_K, 3) arrays from vesin output
# ---------------------------------------------------------------------------

def _build_padded_arrays(
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    D_arr: np.ndarray,
    dist_arr: np.ndarray,
    n_atoms: int,
    max_neighbors: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded unit-vector and mask arrays from a flat neighbor list.

    Parameters
    ----------
    i_arr, j_arr : (N_pairs,) int
    D_arr        : (N_pairs, 3) float  — displacement vectors (not yet unit)
    dist_arr     : (N_pairs,) float    — distances
    n_atoms      : int
    max_neighbors: int

    Returns
    -------
    vectors_padded : (n_atoms, max_neighbors, 3) float32  — unit vectors
    mask_padded    : (n_atoms, max_neighbors) int32
    """
    vectors_padded = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float32)
    mask_padded = np.zeros((n_atoms, max_neighbors), dtype=np.int32)

    # Group by central atom
    counts = np.zeros(n_atoms, dtype=int)
    for k in range(len(i_arr)):
        i = i_arr[k]
        d = dist_arr[k]
        if d < 1e-10:
            continue
        slot = counts[i]
        if slot >= max_neighbors:
            continue
        vectors_padded[i, slot] = (D_arr[k] / d).astype(np.float32)
        mask_padded[i, slot] = 1
        counts[i] += 1

    return vectors_padded, mask_padded


# ---------------------------------------------------------------------------
# Validation: __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from scipy.special import sph_harm_y

    np.random.seed(12345)

    N_atoms = 200
    N_neigh = 12
    MAX_K = 16

    # Random unit vectors
    raw = np.random.randn(N_atoms, N_neigh, 3).astype(np.float32)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True).clip(min=1e-10)
    unit_vecs = raw / norms

    # Build padded arrays (no padding needed here, all slots valid)
    vectors_padded = np.zeros((N_atoms, MAX_K, 3), dtype=np.float32)
    vectors_padded[:, :N_neigh, :] = unit_vecs
    mask_padded = np.zeros((N_atoms, MAX_K), dtype=np.int32)
    mask_padded[:, :N_neigh] = 1

    for l in [4, 6]:
        print(f"\n{'='*60}")
        print(f"l = {l}")
        print(f"{'='*60}")

        # --- scipy reference ---
        def scipy_ql(l, unit_vecs_3d):
            N, K, _ = unit_vecs_3d.shape
            prefactor = 4.0 * np.pi / (2 * l + 1)
            ql_vals = np.zeros(N)
            for i in range(N):
                vecs = unit_vecs_3d[i]
                z_vals = vecs[:, 2]
                theta = np.arccos(np.clip(z_vals, -1.0, 1.0))
                phi = np.arctan2(vecs[:, 1], vecs[:, 0])
                phi = np.where(phi < 0, phi + 2 * np.pi, phi)
                sum_sq = 0.0
                for m in range(-l, l + 1):
                    ylm = sph_harm_y(l, m, theta, phi)
                    qlm = ylm.mean()
                    sum_sq += qlm.real**2 + qlm.imag**2
                ql_vals[i] = np.sqrt(prefactor * sum_sq)
            return ql_vals.astype(np.float32)

        t0 = time.perf_counter()
        ql_scipy = scipy_ql(l, unit_vecs)
        t_scipy = time.perf_counter() - t0
        print(f"scipy:   mean={ql_scipy.mean():.6f}, std={ql_scipy.std():.6f}, "
              f"time={t_scipy*1000:.1f}ms")

        # --- pytorch/numpy fallback ---
        t0 = time.perf_counter()
        ql_pytorch = steinhardt_ql_pytorch(l, vectors_padded, mask_padded)
        t_pytorch = time.perf_counter() - t0
        diff_p = np.abs(ql_scipy - ql_pytorch)
        print(f"pytorch: mean={ql_pytorch.mean():.6f}, std={ql_pytorch.std():.6f}, "
              f"time={t_pytorch*1000:.1f}ms, max_diff={diff_p.max():.2e}")

        # --- warp ---
        if WARP_AVAILABLE:
            t0 = time.perf_counter()
            ql_warp = steinhardt_ql_warp(l, vectors_padded, mask_padded, device="cuda")
            t_warp = time.perf_counter() - t0
            diff_w = np.abs(ql_scipy - ql_warp)
            print(f"warp:    mean={ql_warp.mean():.6f}, std={ql_warp.std():.6f}, "
                  f"time={t_warp*1000:.1f}ms, max_diff={diff_w.max():.2e}")
            tol = 1e-4
            if diff_w.max() < tol:
                print(f"[PASS] warp vs scipy max_abs_diff = {diff_w.max():.2e} < {tol}")
            else:
                print(f"[FAIL] warp vs scipy max_abs_diff = {diff_w.max():.2e} >= {tol}")
        else:
            print("warp: NOT AVAILABLE on this build (CUDA required) — skipping")

        tol = 1e-4
        assert diff_p.max() < tol, (
            f"pytorch vs scipy max diff {diff_p.max():.2e} exceeds tolerance {tol}"
        )
        print(f"[PASS] pytorch vs scipy max_abs_diff = {diff_p.max():.2e} < {tol}")
