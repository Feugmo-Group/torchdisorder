import torch
from torch.autograd.functional import hessian
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch_sim.state import DeformGradMixin, SimState
# Chi-squared loss
from torchdisorder.model.xrd import XRDModel

import logging


logger = logging.getLogger(__name__)

def chi_squared(estimate: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor | float) -> torch.Tensor:
    if isinstance(uncertainty, (float, int)):
        uncertainty = torch.tensor(uncertainty, device=estimate.device, dtype=estimate.dtype)
    return torch.sum((estimate - target) ** 2 / (uncertainty ** 2))

def evaluate(func, x: torch.Tensor):
    """
    Evaluates gradient of a scalar-valued function with respect to 3D Cartesian coordinates.
    Args:
        func: Callable mapping flattened tensor (N_atoms * 3,) -> scalar tensor.
        x: Flattened tensor of shape (N_atoms * 3,).
    Returns:
        Gradient tensor of shape (N_atoms * 3,).
    """
    x = x.clone().detach().requires_grad_(True)
    y = func(x)
    y.backward()
    return x.grad.detach()


def criteria(func, current, next, epsilon, rule):
    """
    Stopping criteria for quasi-Newton optimization.
    Args:
        func: The objective function.
        current: Current flattened position tensor.
        next: Proposed next flattened position tensor.
        epsilon: Threshold for convergence.
        rule: One of 'gradNorm', 'valueDiff', or 'stepLen'.
    """
    if rule == "gradNorm":
        grad_next = evaluate(func, next)
        return torch.norm(grad_next) >= epsilon
    elif rule == "valueDiff":
        return (func(current) - func(next)) >= epsilon
    elif rule == "stepLen":
        return torch.norm(current - next) >= epsilon
    else:
        raise ValueError(f"Unknown stopping rule: {rule}")


def isOutlier(x: torch.Tensor):
    """Raises an error if any NaN or Inf values are present in the tensor."""
    if torch.isnan(x).any():
        raise ValueError("Encountered NaN in tensor.")
    if torch.isinf(x).any():
        raise ValueError("Encountered Inf in tensor.")


class exact:
    def __init__(self, func):
        self.func = func
        self.a = 0
        self.b = 10

    def search(self, x, d, step=0.1, epsilon=1e-6):
        l, r = self.backforth(x, d, self.a, self.b, step)
        return self.gss(x, d, l, r, epsilon)

    def gss(self, x, d, a, b, epsilon):
        ratio = 0.618
        def phi(alpha): return self.func(x + alpha * d)

        t1 = a + (1 - ratio) * (b - a)
        t2 = a + ratio * (b - a)

        while b - a > epsilon:
            if phi(t1) < phi(t2):
                b = t2
                t2 = t1
                t1 = a + (1 - ratio) * (b - a)
            else:
                a = t1
                t1 = t2
                t2 = a + ratio * (b - a)

        return (a + b) / 2

    def backforth(self, x, d, a, b, step):
        def phi(alpha): return self.func(x + alpha * d)

        a0 = (b + a) / 2
        gamma = 1.2

        a1 = a0 + step
        count = 0
        while phi(a1) < phi(a0) or count == 0:
            a = a0
            if phi(a1) >= phi(a0) and count == 0:
                step = -step
            else:
                step *= gamma
            a0 = a1
            a1 = a0 + step
            count += 1
            if a1 < 0:
                a1 = 0
                break
        return min(a, a1), max(a, a1)

class exact3D:
    def __init__(self, func):
        self.func = func
        self.a = 0
        self.b = 10

    def search(self, x, d, step=0.1, epsilon=1e-6):
        l, r = self.backforth(x, d, self.a, self.b, step)
        return self.gss(x, d, l, r, epsilon)

    def gss(self, x, d, a, b, epsilon):
        ratio = 0.618
        def phi(alpha): return self.func(x + alpha * d)

        t1 = a + (1 - ratio) * (b - a)
        t2 = a + ratio * (b - a)

        while b - a > epsilon:
            if phi(t1) < phi(t2):
                b = t2
                t2 = t1
                t1 = a + (1 - ratio) * (b - a)
            else:
                a = t1
                t1 = t2
                t2 = a + ratio * (b - a)

        return (a + b) / 2

    def backforth(self, x, d, a, b, step):
        def phi(alpha): return self.func(x + alpha * d)

        a0 = (b + a) / 2
        gamma = 1.2
        a1 = a0 + step
        count = 0

        while phi(a1) < phi(a0) or count == 0:
            a = a0
            if phi(a1) >= phi(a0) and count == 0:
                step = -step
            else:
                step *= gamma
            a0 = a1
            a1 = a0 + step
            count += 1
            if a1 < 0:
                a1 = 0
                break
        return min(a, a1), max(a, a1)


def wrap_to_cell(delta, cell):
    """
    Apply minimum image convention for displacement `delta` using periodic cell.

    Parameters
    ----------
    delta : Tensor, shape (B, N, 3)
        Displacement vectors (e.g., x1 - x0).
    cell : Tensor, shape (B, 3, 3)
        Cell matrix for each batch.

    Returns
    -------
    Tensor of shape (B, N, 3), wrapped into periodic cell.
    """
    B, N, _ = delta.shape
    inv_cell = torch.inverse(cell)  # (B, 3, 3)
    # Convert Cartesian to fractional
    frac = torch.einsum("bij,bnj->bni", inv_cell, delta)
    # Wrap into [-0.5, 0.5)
    frac_wrapped = frac - frac.round()
    # Back to Cartesian
    return torch.einsum("bij,bnj->bni", cell, frac_wrapped)


class QuasiNewton3D:
    """
    Quasi-Newton optimizer for 3D atomic position optimization.

    This class implements a Quasi-Newton optimization algorithm tailored to
    optimize atomic coordinates arranged as (N_atoms, 3) tensors. It
    approximates the inverse Hessian matrix of the objective function to
    iteratively find a local minimum.

    The optimizer uses gradient information and updates an approximate Hessian
    inverse, applying line search to determine the step size along the
    descent direction.

    Attributes
    ----------
    func : callable
        Objective function taking positions tensor of shape (N_atoms, 3) and
        returning a scalar loss.
    sc : str
        Stopping criterion. One of:
        - 'gradNorm' : stop when gradient norm < tolerance
        - 'valueDiff' : stop when function value difference < tolerance
        - 'stepLen' : stop when step length < tolerance
    explorer : object
        Line search object implementing search strategy to find step size.

    Methods
    -------
    step(x0, eps=1e-6, max_iter=100)
        Runs the Quasi-Newton optimization starting from initial positions x0.
        Returns the optimized positions tensor.

    evaluate(x)
        Computes the gradient of the objective function at positions x.

    criteria(x0, x1, eps)
        Checks whether stopping criterion is met between consecutive positions.

    diff(x1, x0, g)
        Computes displacement (s) and gradient difference (y) vectors for Hessian update.

    update(x1, x0, g, H)
        Abstract method to update the inverse Hessian approximation (to be implemented by subclass).

    initSeq()
        Initializes storage for position sequence during optimization.

    updateSeq()
        Converts the stored sequence of positions into a stacked tensor.

    Parameters
    ----------
    func : callable
        Objective function mapping positions (N_atoms, 3) to scalar loss.

    ls : str, optional
        Line search method to use ('exact' supported), by default 'exact'.

    sc : str, optional
        Stopping criterion ('gradNorm', 'valueDiff', 'stepLen'), by default 'gradNorm'.

    Usage
    -----
    optimizer = QuasiNewton3D(func=my_loss_function)
    x_opt = optimizer.step(x0)

    Notes
    -----
    - Positions are represented as tensors of shape (N_atoms, 3).
    - Hessian inverse approximation is stored as a 4D tensor with shape (N_atoms, 3, N_atoms, 3).
    - The step direction is computed using tensor contraction with `torch.einsum`.
    - The `update` method must be implemented by subclasses with specific Quasi-Newton update formulas (e.g., BFGS).
    """

    def __init__(self, func, ls="exact", sc="gradNorm"):
        """
        Quasi-Newton optimizer for 3D atomic positions (N_atoms, 3).
        Args:
            func: A callable objective function taking x of shape (N_atoms, 3) and returning a scalar.
            ls: Line search strategy ('exact' supported).
            sc: Stopping criterion ('gradNorm', 'valueDiff', 'stepLen').
        """
        self.func = func
        self.sc = sc

        if ls == "exact":
            self.explorer = exact3D(func)
        else:
            raise ValueError(f"Unsupported line search strategy: {ls}")

    def apply_pbc_fractional(self, positions, cell):
        """
        Wrap atomic positions inside the periodic cell using fractional coordinates.

        Parameters:
        -----------
        positions : torch.Tensor, shape (N_atoms, 3)
            Atomic Cartesian positions.
        cell : torch.Tensor, shape (3, 3) or (1, 3, 3)
            Cell vectors defining the periodic box.

        Returns:
        --------
        wrapped_positions : torch.Tensor, shape (N_atoms, 3)
            Positions wrapped inside the cell.
        """
        if cell.dim() == 3:
            cell = cell.squeeze(0)  # Remove batch dim if present

        cell_inv = torch.inverse(cell)  # (3, 3)

        fractional = torch.matmul(positions, cell_inv.T)  # (N_atoms, 3)
        fractional_wrapped = fractional % 1.0  # wrap to [0,1)
        wrapped_positions = torch.matmul(fractional_wrapped, cell.T)  # back to Cartesian

        return wrapped_positions


    def step(self, x0: torch.Tensor, eps=1e-6, max_iter=100):
        """
        Optimize atomic positions directly in 3D (N_atoms, 3).
        Returns the optimized position tensor.
        """
        self.initSeq()

        x0 = x0.clone().detach().requires_grad_(True)
        H = torch.eye(x0.numel(), device=x0.device).reshape(x0.shape + x0.shape)  # (N, 3, N, 3)

        g = self.evaluate(x0)  # shape: (N_atoms, 3)
        self.sequence.append(x0.detach())

        d = -torch.einsum("ijkl,kl->ij", H, g)  # matrix-vector product in 3D
        alpha = self.explorer.search(x0, d)
        x1 = x0 + alpha * d
        isOutlier(x1)

        iter_count = 0
        while self.criteria(x0, x1, eps) and iter_count < max_iter:
            H = self.update(x1, x0, g, H)
            x0 = x1.detach().requires_grad_(True)
            self.sequence.append(x0.detach())

            g = self.evaluate(x0)
            d = -torch.einsum("ijkl,kl->ij", H, g)
            alpha = self.explorer.search(x0, d)
            x1 = x0 + alpha * d
            isOutlier(x1)

            iter_count += 1
        # x1 = self.apply_pbc_fractional(x1, cell)
        self.sequence.append(x1.detach())
        self.updateSeq()
        return x1 # apply_pbc_fractional(x1, cell)

    def evaluate(self, x: torch.Tensor):
        """Compute gradient ∇f(x), shape (N_atoms, 3)"""
        x = x.clone().detach().requires_grad_(True)
        y = self.func(x)
        y.backward()
        return x.grad.detach()

    def criteria(self, x0, x1, eps):
        if self.sc == "gradNorm":
            return torch.norm(self.evaluate(x1)) >= eps
        elif self.sc == "valueDiff":
            return (self.func(x0) - self.func(x1)) >= eps
        elif self.sc == "stepLen":
            return torch.norm(x1 - x0) >= eps
        else:
            raise ValueError(f"Unknown stopping criterion: {self.sc}")

    def diff(self, x1, x0, g):
        g1 = self.evaluate(x1)
        s = x1 - x0
        y = g1 - g
        return s, y  # Both shape (N_atoms, 3)

    def is_vector_nan(self, x):
        return torch.isnan(x).any() or torch.isinf(x).any()

    def update(self, x1, x0, g, H):
        raise NotImplementedError("Must implement in subclass.")

    def initSeq(self):
        self.sequence = []

    def updateSeq(self):
        self.sequence = torch.stack(self.sequence)


#
# class QuasiNewton_2:
#     def __init__(self, func):
#         self.func = func
#
#     def step(self, x: torch.Tensor, eps=1e-6, max_iter=100):
#         """
#         Quasi-Newton step for 3D Cartesian tensor of shape (N_atoms, 3).
#         """
#         N = x.numel()
#         x = x.clone().detach().requires_grad_(True)
#         H = torch.eye(N, dtype=x.dtype, device=x.device)
#
#         for _ in range(max_iter):
#             grad = evaluate(self.func, x).view(-1, 1)  # Shape: (3N, 1)
#             if torch.norm(grad) < eps:
#                 break
#
#             d = -H @ grad  # Shape: (3N, 1)
#             d = d.view_as(x)  # Shape: (N_atoms, 3)
#
#             alpha = gss(self.func, x, d, 0.0, 2.0, 1e-4)
#             x_new = (x + alpha * d).clone().detach().requires_grad_(True)
#
#             grad_new = evaluate(self.func, x_new).view(-1, 1)
#             s = (x_new.view(-1, 1) - x.view(-1, 1))  # (3N, 1)
#             y = grad_new - grad  # (3N, 1)
#
#             H = self.update(H, s, y)
#             x = x_new
#
#         return x.detach()


class BFGS(QuasiNewton3D):
    """
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) Quasi-Newton optimizer for 3D atomic positions.

    This class implements the BFGS update formula to iteratively improve the inverse Hessian
    approximation during optimization of atomic coordinates.

    The update uses displacement and gradient difference vectors to maintain a positive-definite
    Hessian approximation, balancing curvature information with numerical stability.

    Update formula:

        H_new = H
                + (1 + (y^T H y) / (y^T s)) * (s s^T) / (y^T s)
                - (H y s^T + s y^T H) / (y^T s)

    where:
    - H: current inverse Hessian approximation (tensor)
    - s: displacement vector (x1 - x0)
    - y: gradient difference vector (grad(x1) - grad(x0))

    Methods
    -------
    update(x1, x0, g, H)
        Perform a BFGS update on the inverse Hessian approximation.

    Parameters
    ----------
    func : callable
        Objective function to minimize.

    ls : str, optional
        Line search strategy, default is "exact".

    sc : str, optional
        Stopping criterion, default is "gradNorm".
    """

    def __init__(self, func, ls="exact", sc="gradNorm"):
        super().__init__(func, ls, sc)

    def update(self, x1, x0, g, H):
        s = (x1 - x0).reshape(-1, 3, 1)  # (N, 3, 1)
        y = (evaluate(self.func, x1) - g).reshape(-1, 3, 1)  # (N, 3, 1)

        # Transpose for inner products
        sT = s.transpose(1, 2)  # (N, 1, 3)
        yT = y.transpose(1, 2)

        ys = torch.bmm(yT, s)  # (N, 1, 1)
        Hy = torch.bmm(H, y)
        yHy = torch.bmm(yT, Hy)

        term1 = (1 + yHy / ys) * torch.bmm(s, sT) / ys
        term2 = torch.bmm(s, yT)
        term3 = torch.bmm(Hy, sT)
        update = term1 - term2 - term3

        H_new = H + update
        return H_new

class LBFGS3D(QuasiNewton3D):
    """
    Batched, periodic-aware L-BFGS optimizer for 3D systems.

    Attributes
    ----------
    m : int
        Number of (s, y) pairs to store (memory size).
    s_list : list[Tensor] ->  x_{k+1} - x_k,
        Displacement vectors, shape (B, N, 3) per entry.
    y_list : list[Tensor] -> ∇f_{k+1} - ∇f_k
        Gradient differences, shape (B, N, 3) per entry.
    rho_list : list[Tensor]
        Scaling factors (1 / y·s) per batch, shape (B,).

    Usage
    ----------
    lbfgs = LBFGS3D(func=my_grad_fn, m=7, cell=my_cell_tensor)
    x_opt = lbfgs.step(x0)

    """

    def __init__(self, func, m=10, ls="exact", sc="gradNorm", cell=None):
        super().__init__(func, ls, sc)
        self.m = m
        self.cell = cell  # Tensor of shape (B, 3, 3)
        self.s_list = []
        self.y_list = []
        self.rho_list = []

    def update(self, x1, x0, g0, H):
        """
        Update history with new (s, y) pairs.

        Parameters
        ----------
        x1, x0 : Tensor, shape (B, N, 3)
            Current and previous positions.
        g0 : Tensor, shape (B, N, 3)
            Previous gradient.
        H : Dummy argument, kept for interface compatibility.

        Returns
        -------
        Search direction: Tensor of shape (B, N, 3)
        """
        s = x1 - x0
        if self.cell is not None:
            s = wrap_to_cell(s, self.cell)

        y = self.evaluate(x1) - g0

        s_dot_y = torch.sum(s * y, dim=(-1, -2))  # shape (B,)
        mask = s_dot_y > 1e-10  # avoid divide-by-zero

        rho = torch.zeros_like(s_dot_y)
        rho[mask] = 1.0 / s_dot_y[mask]

        if len(self.s_list) == self.m:
            self.s_list.pop(0)
            self.y_list.pop(0)
            self.rho_list.pop(0)

        self.s_list.append(s)
        self.y_list.append(y)
        self.rho_list.append(rho)

        return self.two_loop_recursion(self.evaluate(x1))

    def two_loop_recursion(self, grad):
        """
        Compute the L-BFGS direction using two-loop recursion.

        Parameters
        ----------
        grad : Tensor, shape (B, N, 3)

        Returns
        -------
        Tensor of shape (B, N, 3)
        """
        B, N, _ = grad.shape
        q = grad.clone()
        alpha_list = []

        for s, y, rho in reversed(list(zip(self.s_list, self.y_list, self.rho_list))):
            alpha = rho.view(B, 1, 1) * torch.sum(s * q, dim=(-1, -2), keepdim=True)
            q = q - alpha * y
            alpha_list.append(alpha)

        if self.y_list:
            s = self.s_list[-1]
            y = self.y_list[-1]
            s_dot_y = torch.sum(s * y, dim=(-1, -2), keepdim=True)
            y_dot_y = torch.sum(y * y, dim=(-1, -2), keepdim=True)
            scale = s_dot_y / y_dot_y
            r = scale * q
        else:
            r = q

        for s, y, rho, alpha in zip(self.s_list, self.y_list, self.rho_list, reversed(alpha_list)):
            beta = rho.view(B, 1, 1) * torch.sum(y * r, dim=(-1, -2), keepdim=True)
            r = r + s * (alpha - beta)

        return -r

class DFP(QuasiNewton3D):
    """
    DFP (Davidon-Fletcher-Powell) Quasi-Newton optimizer for 3D atomic positions.

    This class implements the DFP update formula to iteratively improve the inverse Hessian
    approximation used for optimization.

    The update modifies the Hessian approximation by adding curvature information based on
    displacement and gradient differences.

    Update formula:

        H_new = H
                + (s s^T) / (y^T s)
                - (H y y^T H) / (y^T H y)

    where:
    - H: current inverse Hessian approximation (tensor)
    - s: displacement vector (x1 - x0)
    - y: gradient difference vector (grad(x1) - grad(x0))

    Methods
    -------
    update(x1, x0, g, H)
        Perform a DFP update on the inverse Hessian approximation.

    Parameters
    ----------
    func : callable
        Objective function to minimize.

    ls : str, optional
        Line search strategy, default is "exact".

    sc : str, optional
        Stopping criterion, default is "gradNorm".
    """

    def __init__(self, func, ls="exact", sc="gradNorm"):
        super().__init__(func, ls, sc)

    def update(self, x1, x0, g, H):
        s = (x1 - x0).reshape(-1, 3, 1)
        y = (evaluate(self.func, x1) - g).reshape(-1, 3, 1)

        sT = s.transpose(1, 2)
        yT = y.transpose(1, 2)

        Hy = torch.bmm(H, y)

        ssT = torch.bmm(s, sT)
        ys = torch.bmm(yT, s)

        HyHyT = torch.bmm(Hy, Hy.transpose(1, 2))
        yHy = torch.bmm(yT, Hy)

        H_new = H + ssT / ys - HyHyT / yHy
        return H_new


# class DFP(QuasiNewton):
#     def update(self, H, s, y):
#         sTy = s.T @ y
#         if sTy.abs().item() < 1e-12:
#             return H
#         Hy = H @ y
#         term1 = (s @ s.T) / sTy
#         term2 = (Hy @ Hy.T) / (y.T @ Hy + 1e-12)
#         return H + term1 - term2


# class BFGS(QuasiNewton):
#     def update(self, H, s, y):
#         yTs = y.T @ s
#         if yTs.abs().item() < 1e-12:
#             return H
#         rho = 1.0 / yTs
#         I = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
#         V = I - rho * s @ y.T
#         return V @ H @ V.T + rho * s @ s.T

class Broyden(QuasiNewton3D):
    """
    Broyden family Quasi-Newton optimizer for 3D atomic positions.

    This optimizer implements a linear combination of the DFP and BFGS
    inverse Hessian update formulas, controlled by the parameter `phi`.

    The update formula is:

        H_new = (1 - phi) * H_DFP + phi * H_BFGS

    where:
    - H_DFP: inverse Hessian approximation updated by DFP formula
    - H_BFGS: inverse Hessian approximation updated by BFGS formula
    - phi: weighting parameter (0 <= phi <= 1) balancing the two updates

    Attributes
    ----------
    phi : float
        Weighting factor for the BFGS update in the combined formula.
    dfp : DFP
        Instance of the DFP updater.
    bfgs : BFGS
        Instance of the BFGS updater.

    Methods
    -------
    update(x1, x0, g, H)
        Compute the combined Broyden update for the inverse Hessian approximation.

    step(x0, H_init=None, eps=1e-6, max_iter=100)
        Run the optimization starting from initial positions x0.

    Parameters
    ----------
    func : callable
        Objective function to minimize, accepting positions tensor (N_atoms, 3).
    phi : float, optional
        Weight for BFGS update (default 0.5).
    ls : str, optional
        Line search strategy (default 'exact').
    sc : str, optional
        Stopping criterion (default 'gradNorm').
    """

    def __init__(self, func, phi=0.5, ls="exact", sc="gradNorm"):
        super().__init__(func, ls, sc)
        self.phi = phi
        self.dfp = DFP(func)
        self.bfgs = BFGS(func)

    def update(self, x1, x0, g, H):
        # x0, x1 shape: (N, 3)
        # Flatten positions to shape (3N, 1)
        s = (x1 - x0).reshape(-1, 1)
        y = (g(x1) - g(x0)).reshape(-1, 1)

        H_DFP = self.dfp.update(s, y, H)
        H_BFGS = self.bfgs.update(s, y, H)

        return (1 - self.phi) * H_DFP + self.phi * H_BFGS

    def step(self, x0, H_init=None, eps=1e-6, max_iter=100):
        return super().step(x0, H_init=H_init, eps=eps, max_iter=max_iter)




@dataclass
class AugLagState(SimState):
    loss: Optional[torch.Tensor] = None
    G_r: Optional[torch.Tensor] = None
    T_r: Optional[torch.Tensor] = None
    S_Q: Optional[torch.Tensor] = None
    q_tet: Optional[torch.Tensor] = None
    diagnostics: Optional[dict] = None
    system_idx: Optional[torch.Tensor] = None
    n_systems:Optional[torch.Tensor] = None

import torch
from torch import nn


class AugmentedLagrangian(nn.Module):
    def __init__(self, objective, model, lam, sigma, constraints_eq=None,constraints_ineq=None, method="BFGS", phi=0.5, ls="exact", sc="gradNorm", device=None, dtype=torch.float32):
        super().__init__()
        self.objective = objective(model)
        self.model = model
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.equality= constraints_eq
        self.inequality = constraints_ineq

        self.lam = nn.Parameter(torch.tensor(lam, dtype=dtype, device=device), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=dtype, device=device), requires_grad=False)
        self.eta = self.lam / self.sigma

        self.method = method
        self.phi = phi
        self.ls = ls
        self.sc = sc

        self._base_state = None
        self.out = None
        self.lag = None
        self.eps = None
        self.max_iter = None

    def forward(self, state: AugLagState):
        self._base_state = state
        return self._update_state(state.positions)

    def step(self, state: AugLagState, eps=1e-6, max_iter=100):
        #x = state.positions.clone().detach().requires_grad_(True)
        x = state.positions


        if self.method == "BFGS":
            solver = BFGS(self.aug_func, ls=self.ls, sc=self.sc)
        elif self.method == "DFP":
            solver = DFP(self.aug_func, ls=self.ls, sc=self.sc)
        elif self.method == "Broyden":
            solver = Broyden(self.aug_func, phi=self.phi, ls=self.ls, sc=self.sc)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        x = solver.step(x, eps=eps, max_iter=max_iter)
        updated_state = self._update_state(x)
        self.update_lagrange_multipliers(updated_state)
        self.sigma.data *= 10
        self.eta = self.lam / self.sigma
        return updated_state

    def aug_func(self, x: torch.Tensor) -> torch.Tensor:
        state = self._update_state(x)
        lag = self.objective(state)

        for i, c in enumerate(self.equality):
            val = c(state)
            lag += -self.lam[i] * val + 0.5 * self.sigma[i] * val ** 2

        for j, (f, is_ge) in enumerate(self.inequality):
            i = j + len(self.equality)
            phi = f(state)
            if is_ge:
                val = torch.min(phi - self.eta[i], torch.tensor(0.0, device=x.device, dtype=self.dtype))
            else:
                val = torch.min(-phi - self.eta[i], torch.tensor(0.0, device=x.device, dtype=self.dtype))
            lag += 0.5 * self.sigma[i] * (val ** 2 - self.eta[i] ** 2)

        state.loss = lag
        self.lag = lag
        return lag

    def update_lagrange_multipliers(self, state: AugLagState):
        for i, c in enumerate(self.equality):
            self.lam.data[i] -= self.sigma[i] * c(state).detach()

        for j, (f, is_ge) in enumerate(self.inequality):
            i = j + len(self.equality)
            phi = f(state)
            if is_ge:
                self.lam.data[i] = -self.sigma[i] * torch.min(phi - self.eta[i], torch.tensor(0.0, device=phi.device, dtype=self.dtype)).detach()
            else:
                self.lam.data[i] = -self.sigma[i] * torch.min(-phi - self.eta[i], torch.tensor(0.0, device=phi.device, dtype=self.dtype)).detach()
        self.eta = self.lam / self.sigma

    def kkt_residual(self, state: AugLagState):
        val = 0.0
        for c in self.equality:
            val += c(state).pow(2)
        for j, (f, is_ge) in enumerate(self.inequality):
            i = j + len(self.equality)
            phi = f(state)
            if is_ge:
                val += torch.min(phi, self.eta[i]).pow(2)
            else:
                val += torch.min(-phi, self.eta[i]).pow(2)
        return val.sqrt()

    def _update_state(self, positions: torch.Tensor) -> AugLagState:
        state = self._base_state
        state.positions = positions

        results = self.model(state)

        return AugLagState(
            positions=positions,
            masses=state.masses,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
            pbc=state.pbc,
            loss=self.lag,
            G_r=results.get("G_r", None),
            T_r=results.get("T_r", None),
            S_Q=results.get("S_Q", None),
            q_tet=results.get("q_tet", None),
            diagnostics=results,
        )
class AugLagNNN(nn.Module):
    def __init__(self, objective, model, lam, sigma, constraints_eq=None, constraints_ineq=None,
                 method="BFGS", phi=0.5, ls="exact", sc="gradNorm", device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.model = model
        self.objective = objective(model)
        self.equality = constraints_eq or []
        self.inequality = [
            (c, False) if not isinstance(c, tuple) else c
            for c in (constraints_ineq or [])
        ]

        # Assume lam and sigma are initialized as [n_systems, n_constraints]
        self.lam = nn.Parameter(torch.tensor(lam, dtype=dtype, device=self.device), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=dtype, device=self.device), requires_grad=False)
        self.eta = self.lam / self.sigma

        self.method = method
        self.phi = phi
        self.ls = ls
        self.sc = sc

        self._base_state = None
        self.out = None
        self.lag = None
        self.eps = None
        self.max_iter = None

        self.plot = False
        self.solver = None

    def _update_state(self, positions: torch.Tensor):
        state = self._base_state
        state.positions = positions

        results = self.model(state)

        return AugLagState(
            positions=positions,
            masses=state.masses,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
            pbc=state.pbc,
            loss=self.lag,
            G_r=results.get("G_r", None),
            T_r=results.get("T_r", None),
            S_Q=results.get("S_Q", None),
            q_tet=results.get("q_tet", None),
            diagnostics=results,
        )

    def forward(self, state):
        self._base_state = state
        return self._update_state(state.positions)

    def aug_func(self, x):
        state = self._update_state(x)
        base_loss = self.objective(state)
        lag = base_loss.clone()

        for b in range(state.n_systems):
            mask = state.system_idx == b
            sub_state = state.masked_select(mask)

            for i, c in enumerate(self.equality):
                val = c(sub_state)
                lag += -self.lam[b, i] * val + 0.5 * self.sigma[b, i] * val ** 2

            for j, (f, is_ge) in enumerate(self.inequality):
                i = j + len(self.equality)
                phi = f(sub_state)
                if is_ge:
                    val = torch.min(phi - self.eta[b, i], torch.tensor(0.0, device=self.device, dtype=self.dtype))
                else:
                    val = torch.min(-phi - self.eta[b, i], torch.tensor(0.0, device=self.device, dtype=self.dtype))
                lag += 0.5 * self.sigma[b, i] * (val ** 2 - self.eta[b, i] ** 2)

        state.loss = lag
        self.lag = lag
        return lag

    def step(self, state: AugLagState, eps=1e-6, max_iter=100, epsilon=1e-6):
        self._base_state = state
        x = state.positions

        if self.method == "BFGS":
            self.solver = BFGS(self.aug_func, ls=self.ls, sc=self.sc)
        elif self.method == "DFP":
            self.solver = DFP(self.aug_func, ls=self.ls, sc=self.sc)
        elif self.method == "Broyden":
            self.solver = Broyden(self.aug_func, phi=self.phi, ls=self.ls, sc=self.sc)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        assert self.isFeasible(state), "Initial point isn't feasible"
        count = 1
        print(f"round {count}:")

        x = self.solver.step(x, eps=eps ,max_iter=max_iter)

        updated_state = self._update_state(x)
        self.update_lam(updated_state)

        while self.criteria(updated_state) > epsilon:
            self.sigma.data *= 10
            self.update_eta()
            count += 1
            print(f"round {count}:")

            x = self.solver.step(updated_state.positions, eps=eps ,max_iter=max_iter)
            updated_state = self._update_state(x)
            self.update_lam(updated_state)

        return updated_state

    def isFeasible(self, state):
        flag = True
        for b in range(state.n_systems):
            mask = state.system_idx == b
            sub_state = state.masked_select(mask)

            for eq in self.equality:
                if not torch.allclose(eq(sub_state), torch.tensor(0., dtype=self.dtype, device=self.device), atol=1e-6):
                    flag = False
            for neq, is_pos in self.inequality:
                val = neq(sub_state)
                if is_pos:
                    if not (val > 0).all():
                        flag = False
                else:
                    if not (-val > 0).all():
                        flag = False
        return flag

    def update_lam(self, state):
        for b in range(state.n_systems):
            mask = state.system_idx == b
            sub_state = state.masked_select(mask)

            for i, eq in enumerate(self.equality):
                self.lam.data[b, i] -= self.sigma[b, i] * eq(sub_state).detach()

            for j, (neq, is_pos) in enumerate(self.inequality):
                i = j + len(self.equality)
                val = neq(sub_state)
                if is_pos:
                    self.lam.data[b, i] = -self.sigma[b, i] * torch.min(val - self.eta[b, i], torch.tensor(0., device=self.device)).detach()
                else:
                    self.lam.data[b, i] = -self.sigma[b, i] * torch.min(-val - self.eta[b, i], torch.tensor(0., device=self.device)).detach()

    def update_eta(self):
        self.eta.data = self.lam.data / self.sigma

    def criteria(self, state):
        val = 0.0
        for b in range(state.n_systems):
            mask = state.system_idx == b
            sub_state = state.masked_select(mask)

            for i, eq in enumerate(self.equality):
                val += eq(sub_state).pow(2).sum()

            for j, (neq, is_pos) in enumerate(self.inequality):
                i = j + len(self.equality)
                phi = neq(sub_state)
                if is_pos:
                    val += torch.min(phi, self.eta[b, i]).pow(2).sum()
                else:
                    val += torch.min(-phi, self.eta[b, i]).pow(2).sum()

        return val.sqrt()



class AugLagNN(nn.Module):
    def __init__(
            self,
            objective,
            model,
            lam,
            sigma,
            eta,
            mask=None,
            constraints_eq=None,
            constraints_ineq=None,
            method="BFGS",
            phi=0.5,
            ls="exact",
            sc="gradNorm",
            device=None,
            dtype=torch.float32,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.model = model
        self.objective = objective(model)
        self.equality = constraints_eq or []
        self.inequality = constraints_ineq
        # lam and sigma shape: (n_systems, max_atoms)
        self.lam = nn.Parameter(torch.tensor(lam, dtype=dtype, device=self.device), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=dtype, device=self.device), requires_grad=False)
        self.eta = eta
        self.mask = mask
        self.method = method
        self.phi = phi
        self.ls = ls
        self.sc = sc

        self._base_state = None
        self.out = None
        self.lag = None
        self.eps = None
        self.max_iter = None

        self.plot = False
        self.solver = None

    def _update_state(self, positions: torch.Tensor):

        state = self._base_state
        cell = state.cell  # [n_systems, 3, 3]
        pbc = state.pbc  # [n_systems, 3]
        system_idx = state.system_idx
        pos = positions.clone()
        # Wrap positions into the unit cell
        if pbc:
            with torch.no_grad():
                for b in range(state.n_systems):
                    mask = system_idx == b
                    pos_b = pos[mask]  # [n_atoms_b, 3]
                    cell_b = cell[b]  # [3, 3]
                    inv_cell_b = torch.inverse(cell_b)  # [3, 3]

                    # Convert to fractional coordinates
                    frac_b = torch.matmul(pos_b, inv_cell_b.T)

                    # Apply PBC: wrap fractional coordinates to [0, 1)
                    frac_b = frac_b % 1.0

                    # Convert back to Cartesian
                    pos[mask] = torch.matmul(frac_b, cell_b)

            pos.requires_grad_(True)

        # Update base state positions
        state.positions = positions

        results = self.model(state)

        return AugLagState(
            positions=positions,
            masses=state.masses,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
            pbc=state.pbc,
            loss=self.lag,
            G_r=results.get("G_r", None),
            T_r=results.get("T_r", None),
            S_Q=results.get("S_Q", None),
            q_tet=results.get("q_tet", None),
            diagnostics=results,
        )

    def forward(self, state):
        self._base_state = state
        return self._update_state(state.positions)

    def aug_func(self, x):
        """
            Evaluate the augmented Lagrangian function for a given input `x`.

            This function computes the total loss composed of:
            - The base objective function `f(x)`
            - Penalty and Lagrange multiplier terms for equality constraints: c_i(x) = 0
            - Smoothed penalty terms for inequality constraints:
              - f_j(x) >= 0 (if `is_ge` is True)
              - f_j(x) <= 0 (if `is_ge` is False)

            The augmented Lagrangian has the following structure:

                L_aug(x) = f(x)
                          + Σ_i [ -λ_i c_i(x) + 0.5 * σ_i * c_i(x)^2 ]
                          + Σ_j [ -λ_j φ_j(x) + 0.5 * σ_j (φ_j(x)^2 - η_j^2) ]

            where:
            - c_i(x): equality constraint functions
            - f_j(x): inequality constraint functions
            - φ_j(x): smoothed constraint violation:
                - φ_j(x) = min(f_j(x) - η_j, 0) if f_j(x) ≥ 0
                - φ_j(x) = min(-f_j(x) - η_j, 0) if f_j(x) ≤ 0
            - λ: Lagrange multipliers
            - σ: penalty coefficients
            - η: slack-like variables (for inequalities)

            Masking is applied to allow constraint selection or per-system handling.

            Parameters
            ----------
            x : torch.Tensor
                The input optimization variable (usually flattened). It will be transformed into a `state` object internally.

            Returns
            -------
            torch.Tensor
                The scalar value of the augmented Lagrangian loss at the given input `x`. This can be minimized using any optimizer.
            """
        state = self._update_state(x)
        base_loss = self.objective(state)
        lag = base_loss.clone()

        # Mask the per-atom parts (excluding index 0, which is for equality)
        lam_masked = self.lam[self.mask]
        eta_masked = self.eta[self.mask]
        sigma_masked = self.sigma[self.mask]


        for i, c in enumerate(self.equality):
            val_eq = c(state)
            lag += -lam_masked * val_eq + 0.5 * sigma_masked * val_eq ** 2

        for j, (f, is_ge) in enumerate(self.inequality):
            #i = j + len(self.equality)
            phi = f(state)
            if is_ge:
                val = torch.min(phi - eta_masked, torch.zeros_like(phi))
            else:
                val = torch.min(-phi - eta_masked, torch.zeros_like(phi))
            #lag += torch.sum(-lam_masked * val + 0.5 * sigma_masked * (val ** 2 - eta_masked ** 2))
            lag += torch.sum(-lam_masked * val + 0.5 * sigma_masked * torch.norm(val - eta_masked, dim=-1)**2)


        state.loss = lag
        self.lag = lag
        return lag





    def step(self, state: 'AugLagState', eps=1e-6, max_iter=100, epsilon=1e-6):
        self._base_state = state
        if self.method == "LBFGS3D":
            self.solver = LBFGS3D(self.aug_func, m=10, ls=self.ls, sc=self.sc, cell=state.cell)
        if self.method == "BFGS":
            self.solver = BFGS(self.aug_func, ls=self.ls, sc=self.sc)
        elif self.method == "DFP":
            self.solver = DFP(self.aug_func, ls=self.ls, sc=self.sc)
        elif self.method == "Broyden":
            self.solver = Broyden(self.aug_func, phi=self.phi, ls=self.ls, sc=self.sc)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        assert self.isFeasible(state), "Initial point isn't feasible"
        count = 1
        logger.info(f"round {count}:")

        x = self.solver.step(state.positions, eps=eps, max_iter=max_iter)

        updated_state = self._update_state(x)
        self.update_lam(updated_state)

        while self.criteria(updated_state) > epsilon:
            self.sigma.data *= 10
            self.update_eta()
            count += 1
            logger.info(f"round {count}:")

            x = self.solver.step(updated_state.positions, eps=eps, max_iter=max_iter)
            updated_state = self._update_state(x)
            self.update_lam(updated_state)

        return updated_state

    def isFeasible(self, state):

        """
            Check whether the current state satisfies all equality and inequality constraints.

            This function evaluates:
            - Equality constraints c_i(state) ≈ 0, using a numerical tolerance
            - Inequality constraints:
                - f_j(state) ≥ 0 if `is_pos` is True
                - f_j(state) ≤ 0 if `is_pos` is False

            A state is considered feasible if:
            - All equality constraints are satisfied within a tolerance of 1e-6
            - All inequality constraints are strictly satisfied (i.e., all elements > 0 or < 0 as required)

            Note: The current implementation assumes scalar or elementwise constraints and does not handle batching.

            Parameters
            ----------
            state : Any
                A state object that provides values for the constraints when evaluated.

            Returns
            -------
            bool
                True if all constraints are satisfied, False otherwise.
            """
        flag = True

        for eq in self.equality:
            if not torch.allclose(eq(state), torch.tensor(0., dtype=self.dtype, device=self.device), atol=1e-6):
                flag = False

        for neq, is_pos in self.inequality:
            #neq, is_pos  = self.inequality
            val = neq(state)
            if is_pos:
                if not (val > 0).all():
                    flag = False
            else:
                if not (-val > 0).all():
                    flag = False

            return flag

    def update_lam(self, state):
        """
            Update Lagrange multipliers (`lam`) for both equality and inequality constraints
            based on the current state.

            For equality constraints c_i(x) = 0:
                λ_i ← λ_i - σ_i * c_i(x)

            For inequality constraints:
                If f_j(x) ≥ 0:
                    λ_j ← -σ_j * min(f_j(x) - η_j, 0)
                If f_j(x) ≤ 0:
                    λ_j ← -σ_j * min(-f_j(x) - η_j, 0)

            Only the masked subset of constraints (defined by `self.mask`) is updated.

            Parameters
            ----------
            state : Any
                The current simulation or optimization state used to evaluate constraint values.
            """
        # n_eq = len(self.equality)
        #
        # # Update equality constraints (assumed small number, usually 1)
        # for i, eq in enumerate(self.equality):
        #     # Optional: check mask on equality constraints if mask applies here
        #     self.lam[i] -= self.sigma[i] * eq(state).detach()

        # Get masked indices for inequalities (offset by n_eq)
        masked_indices = torch.nonzero(self.mas, as_tuple=False).flatten()

        for j, (f, is_pos) in enumerate(self.inequality):

            val = f(state) # Get function for this inequality
            for  i, idx in enumerate(masked_indices):

                if is_pos:
                    update_val = -self.sigma[idx] * torch.min(val[i] - self.eta[idx],
                                                            torch.tensor(0., device=self.device)).detach()
                else:
                    update_val = -self.sigma[idx] * torch.min(-val[i] - self.eta[idx],
                                                            torch.tensor(0., device=self.device)).detach()

                self.lam[idx] = update_val

    def update_eta(self):
        """
            Update slack-like variables (`eta`) for inequality constraints based on the current
            values of Lagrange multipliers and penalty coefficients.

            For masked inequality constraints:
                η_j ← λ_j / σ_j

            This update enables smooth handling of inequality constraint violations in the
            augmented Lagrangian formulation.

            Note
            ----
            Only constraints selected by `self.mask` are updated.
            """
        self.eta[self.mask] = self.lam[self.mask] / self.sigma[self.mask]

    def criteria(self, state):
        """
            Compute the feasibility violation norm of all constraints at the given state.

            The result is the root of the sum of squared residuals:
                - For equality constraints: ∑ ||c_i(x)||²
                - For inequality constraints:
                    - If f_j(x) ≥ 0: ∑ ||min(f_j(x), η_j)||²
                    - If f_j(x) ≤ 0: ∑ ||min(-f_j(x), η_j)||²

            This metric serves as a convergence or feasibility check during optimization.

            Parameters
            ----------
            state : Any
                The current state from which to evaluate constraint values.

            Returns
            -------
            torch.Tensor
                A scalar value representing the total constraint violation norm.
            """
        val = 0.0

        for i, eq in enumerate(self.equality):
            eq_val = eq(state)
            val += eq_val.pow(2).sum()

        for j, (ineq, is_pos) in enumerate(self.inequality):
            i_idx = j + len(self.equality)
            eta_masked = self.eta[self.mask]
            phi = ineq(state)
            #eta = self.eta[i_idx]  # Only one system (index 0)
            if is_pos:
                val += torch.min(phi,eta_masked).pow(2).sum()
            else:
                val += torch.min(-phi, eta_masked).pow(2).sum()

        return val.sqrt()

# To use:
# model = DummyModel(spec_calc, rdf_data)
# objective, constraints = make_objective_and_constraints(model)
# x0 = torch.randn(num_atoms * 3, dtype=torch.float32)  # initial positions
# aug = AugmentedLagrangian(objective, lam=[0, 0], sigma=[1, 1], constraints_eq=constraints, method="DFP")
# x_opt = aug.optimize(x0)
