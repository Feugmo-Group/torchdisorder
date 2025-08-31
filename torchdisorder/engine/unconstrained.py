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

class LBFGS3D_batch(QuasiNewton3D):
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
        # if self.cell is not None:
        #     s = wrap_to_cell(s, self.cell)

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



class LBFGS3D(QuasiNewton3D):
    """
    Non-batched L-BFGS optimizer for 3D systems.

    Attributes
    ----------
    m : int
        Number of (s, y) pairs to store.
    s_list : list[Tensor] ->  x_{k+1} - x_k, shape (N, 3)
    y_list : list[Tensor] -> ∇f_{k+1} - ∇f_k, shape (N, 3)
    rho_list : list[float] -> 1 / (y · s)

    Usage
    ----------
    lbfgs = LBFGS3D(func=my_grad_fn, m=7)
    x_opt = lbfgs.step(x0)
    """

    def __init__(self, func, m=10, ls="exact", sc="gradNorm", cell=None):
        self.func = func  # Gradient function
        self.m = m  # History size
        self.cell = cell  # Optional periodic cell (3, 3)
        self.s_list = []
        self.y_list = []
        self.rho_list = []

    def evaluate(self, x):
        return self.func(x)

    def update(self, x1, x0, g0, H=None):
        """
        Update history with new (s, y) pairs.

        Parameters
        ----------
        x1, x0 : Tensor, shape (N, 3)
            Current and previous positions.
        g0 : Tensor, shape (N, 3)
            Previous gradient.
        H : unused, for compatibility.

        Returns
        -------
        Search direction: Tensor of shape (N, 3)
        """
        s = x1 - x0
        # if self.cell is not None:
        #     s = wrap_to_cell(s, self.cell)

        g1 = self.evaluate(x1)
        y = g1 - g0

        s_dot_y = torch.sum(s * y)
        if s_dot_y < 1e-10:
            return -g1  # fallback to gradient descent

        rho = 1.0 / s_dot_y

        if len(self.s_list) == self.m:
            self.s_list.pop(0)
            self.y_list.pop(0)
            self.rho_list.pop(0)

        self.s_list.append(s)
        self.y_list.append(y)
        self.rho_list.append(rho)

        return self.two_loop_recursion(g1)

    def two_loop_recursion(self, grad):
        """
        Compute the L-BFGS direction using two-loop recursion.

        Parameters
        ----------
        grad : Tensor, shape (N, 3)

        Returns
        -------
        Tensor of shape (N, 3)
        """
        q = grad.clone()
        alpha_list = []

        for s, y, rho in reversed(list(zip(self.s_list, self.y_list, self.rho_list))):
            alpha = rho * torch.sum(s * q)
            q = q - alpha * y
            alpha_list.append(alpha)

        if self.y_list:
            s = self.s_list[-1]
            y = self.y_list[-1]
            scale = torch.sum(s * y) / torch.sum(y * y)
            r = scale * q
        else:
            r = q

        for s, y, rho, alpha in zip(self.s_list, self.y_list, self.rho_list, reversed(alpha_list)):
            beta = rho * torch.sum(y * r)
            r = r + s * (alpha - beta)

        return -r




@dataclass
class AugLagState(SimState):
    # loss: Optional[torch.Tensor] = None
    G_r: Optional[torch.Tensor] = None
    T_r: Optional[torch.Tensor] = None
    S_Q: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None
    diagnostics: Optional[dict] = None
    system_idx: Optional[torch.Tensor] = None
    n_systems:Optional[torch.Tensor] = None

import torch
from torch import nn


class LBFGSWrapper1:
    def __init__(self, func, positions, max_iter=100, eps=1e-6):
        self.func = func
        self.positions = positions
        self.positions.requires_grad_(True)
        self.optimizer = torch.optim.LBFGS([self.positions], max_iter=max_iter, tolerance_grad=eps)

    def step(self, x, eps=1e-6, max_iter=100):
        # x is ignored here — we already store positions internally
        def closure():
            self.optimizer.zero_grad()
            loss = self.func(self.positions)
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return self.positions





class AugLagNNNN(nn.Module):
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
        method="LBFGS",
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
        self.inequality = constraints_ineq or []

        self.lam = nn.Parameter(torch.tensor(lam, dtype=dtype, device=self.device), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=dtype, device=self.device), requires_grad=False)
        self.eta = eta
        self.mask = mask if mask is not None else torch.ones_like(torch.tensor(lam, dtype=torch.bool))

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
        # if pbc:
        #     with torch.no_grad():
        #         for b in range(state.n_systems):
        #             mask = system_idx == b
        #             pos_b = pos[mask]  # [n_atoms_b, 3]
        #             cell_b = cell[b]  # [3, 3]
        #             inv_cell_b = torch.inverse(cell_b)  # [3, 3]
        #
        #             # Convert to fractional coordinates
        #             frac_b = torch.matmul(pos_b, inv_cell_b.T)
        #
        #             # Apply PBC: wrap fractional coordinates to [0, 1)
        #             frac_b = frac_b % 1.0
        #
        #             # Convert back to Cartesian
        #             pos[mask] = torch.matmul(frac_b, cell_b)
        #
        #     pos.requires_grad_(True)

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
            G_r=results.get("G_r", None),
            T_r=results.get("T_r", None),
            S_Q=results.get("S_Q", None),
            q=results.get("q", None),
            diagnostics={},
        )

    def isFeasible(self, state):
        flag = True
        for eq in self.equality:
            if not torch.allclose(eq(state), torch.tensor(0., dtype=self.dtype, device=self.device), atol=1e-6):
                flag = False

        for neq, is_pos in self.inequality:
            val = neq(state)
            if is_pos:
                if not (val > 0).all():
                    flag = False
            else:
                if not (-val > 0).all():
                    flag = False

        return flag if self.equality or self.inequality else True

    def aug_func(self, x):
        state = self._update_state(x)
        base_loss = self.objective(state)
        lag = base_loss.clone()
        state.diagnostics["chi2"] = lag
        if self.equality:
            lam_masked = self.lam[self.mask]
            sigma_masked = self.sigma[self.mask]
            for i, c in enumerate(self.equality):
                val_eq = c(state)
                lag += -lam_masked * val_eq + 0.5 * sigma_masked * val_eq ** 2

        if self.inequality:
            lam_masked = self.lam[self.mask]
            eta_masked = self.eta[self.mask]
            sigma_masked = self.sigma[self.mask]
            for j, (f, is_ge) in enumerate(self.inequality):
                phi = f(state)
                if is_ge:
                    val = torch.min(phi - eta_masked, torch.zeros_like(phi))
                else:
                    val = torch.min(-phi - eta_masked, torch.zeros_like(phi))
                lag += torch.sum(-lam_masked * val + 0.5 * sigma_masked * torch.norm(val - eta_masked, dim=-1) ** 2)

        state.diagnostics["loss"] = lag
        self.lag = lag
        return lag

    def update_lam(self, state):
        masked_indices = torch.nonzero(self.mask, as_tuple=False).flatten()
        for j, (f, is_pos) in enumerate(self.inequality):
            val = f(state)
            for i, idx in enumerate(masked_indices):
                if is_pos:
                    update_val = -self.sigma[idx] * torch.min(val[i] - self.eta[idx], torch.tensor(0., device=self.device)).detach()
                else:
                    update_val = -self.sigma[idx] * torch.min(-val[i] - self.eta[idx], torch.tensor(0., device=self.device)).detach()
                self.lam[idx] = update_val

    def update_eta(self):
        self.eta[self.mask] = self.lam[self.mask] / self.sigma[self.mask]

    def criteria(self, state):
        val = 0.0
        for eq in self.equality:
            val += eq(state).pow(2).sum()
        for j, (ineq, is_pos) in enumerate(self.inequality):
            eta_masked = self.eta[self.mask]
            phi = ineq(state)
            if is_pos:
                val += torch.min(phi, eta_masked).pow(2).sum()
            else:
                val += torch.min(-phi, eta_masked).pow(2).sum()
        return val.sqrt()

    def forward(self, state):
        self._base_state = state
        return self._update_state(state.positions)


    def step(self, state, eps=1e-6, max_iter=100, epsilon=1e-6):
        self._base_state = state

        if self.method == "LBFGS":
            self.solver = LBFGSWrapper(self.aug_func, state.positions, max_iter=max_iter, eps=eps)
        elif self.method == "LBFGS3D":
            self.solver = LBFGS3D(self.aug_func, m=10, ls=self.ls, sc=self.sc, cell=state.cell)
        elif self.method == "BFGS":
            self.solver = BFGS(self.aug_func, ls=self.ls, sc=self.sc)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        assert self.isFeasible(state), "Initial point isn't feasible"
        x = self.solver.step(state.positions, eps=eps, max_iter=max_iter)
        logging.info(x)
        updated_state = self._update_state(x)

        if self.equality or self.inequality:
            self.update_lam(updated_state)

        count = 1
        while self.equality or self.inequality and self.criteria(updated_state) > epsilon:
            logger.info(f"round {count}:")
            self.sigma.data *= 10
            self.update_eta()
            count += 1
            x = self.solver.step(updated_state.positions, eps=eps, max_iter=max_iter)
            updated_state = self._update_state(x)
            self.update_lam(updated_state)

        return updated_state

# To use:
# model = DummyModel(spec_calc, rdf_data)
# objective, constraints = make_objective_and_constraints(model)
# x0 = torch.randn(num_atoms * 3, dtype=torch.float32)  # initial positions
# aug = AugmentedLagrangian(objective, lam=[0, 0], sigma=[1, 1], constraints_eq=constraints, method="DFP")
# x_opt = aug.optimize(x0)




class AugLagNN:
    def __init__(self, objective, model, constraints_eq=[], constraints_ineq=[],
                 lam=None, sigma=None, eta=None, mask=None, method="LBFGS", max_iter=100, eps=1e-5):
        self.objective = objective(model)
        self.model = model
        self.equality = constraints_eq
        self.inequality = constraints_ineq
        self.method = method
        self.max_iter = max_iter
        self.eps = eps

        # Constraint parameters
        self.lam = lam
        self.sigma = sigma
        self.eta = eta
        self.mask = mask
        self._base_state = None

        # Initialize solver immediately
        if method == "LBFGS":
            self.solver = LBFGSWrapper(self.model , max_iter=max_iter, eps=eps)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    #     self.solver = None  # Will be initialized once state is provided
    #
    def __call__(self, state: AugLagState):
        self._base_state = state

        # Initialize solver after state is known
        if self.method == "LBFGS":
            self.solver = LBFGSWrapper(self.objective,  max_iter=self.max_iter, eps=self.eps)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return state

    def aug_func(self, x):
        state = self._update_state(x)
        #base_loss = self.objective(state)
        #lag = base_loss.clone()
        lag = self.solver.last_loss
        diagnostics = {"loss": lag, "lagrange": {}, "sigma": {}}
        lam_masked = self.lam[self.mask]
        eta_masked = self.eta[self.mask]
        sigma_masked = self.sigma[self.mask]
        for i, c in enumerate(self.equality):
            val = c(state)
            lag += -lam_masked * val + 0.5 * sigma_masked * val_eq ** 2
            #lag += -self.lam[i] * val + 0.5 * self.sigma[i] * val ** 2
            diagnostics["lagrange"][f"eq_{i}"] = val.item()
            diagnostics["sigma"][f"eq_{i}"] = self.sigma[i].item()

        for j, (f, is_ge) in enumerate(self.inequality):
            val = f(state)
            if not is_ge:
                val = -val
            #pos_val = torch.clamp(val, min=-self.eta[j])
            pos_val = torch.clamp(val, min=-eta_masked)
            #lag += -self.lam[len(self.equality) + j] * pos_val + 0.5 * self.sigma[len(self.equality) + j] * pos_val ** 2
            lag += torch.sum(-lam_masked * pos_val + 0.5 * sigma_masked * torch.norm(pos_val - eta_masked, dim=-1) ** 2)
            diagnostics["lagrange"][f"ineq_{j}"] = pos_val.item()
            #diagnostics["sigma"][f"ineq_{j}"] = self.sigma[len(self.equality) + j].item()
            diagnostics["sigma"][f"ineq_{j}"] = sigma_masked.item()


        state.diagnostics = diagnostics
        return lag

    def _update_state(self, positions: torch.Tensor):
        state = self._base_state
        cell = state.cell
        pbc = state.pbc
        system_idx = state.system_idx

        pos = positions.clone()
        state.positions = pos

        results = self.model(state)

        return AugLagState(
            positions=positions,
            masses=state.masses,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
            system_idx=state.system_idx,
            n_systems=state.n_systems,
            pbc=state.pbc,
            G_r=results.get("G_r", None),
            T_r=results.get("T_r", None),
            S_Q=results.get("S_Q", None),
            q=results.get("q", None),
            diagnostics={},
        )

    # def update_lam(self, state: AugLagState):
    #
    #
    #     for i, c in enumerate(self.equality):
    #         val = c(state).detach()
    #         #self.lam[i] -= self.sigma[i] * val
    #
    #     for j, (f, is_ge) in enumerate(self.inequality):
    #         val = f(state).detach()
    #         if not is_ge:
    #             val = -val
    #         idx = len(self.equality) + j
    #         self.lam[idx] = torch.clamp(self.lam[idx] - self.sigma[idx] * val, min=0)

    def update_lam(self, state):
        masked_indices = torch.nonzero(self.mask, as_tuple=False).flatten()
        for j, (f, is_ge) in enumerate(self.inequality):
            val = f(state)
            for i, idx in enumerate(masked_indices):
                if is_ge:
                    update_val = -self.sigma[idx] * torch.min(val[i] - self.eta[idx],
                                                              torch.tensor(0., device=self.device)).detach()
                else:
                    update_val = -self.sigma[idx] * torch.min(-val[i] - self.eta[idx],
                                                              torch.tensor(0., device=self.device)).detach()
                self.lam[idx] = update_val

    def update_eta(self):

        for j in range(len(self.inequality)):
            idx = len(self.equality) + j

            self.eta[j][self.mask] = max(self.eta[j][self.mask], -self.lam[idx][self.mask]/ self.sigma[idx][self.mask])

    def criteria(self, state: AugLagState):
        lam_masked = self.lam[self.mask]
        eta_masked = self.eta[self.mask]
        sigma_masked = self.sigma[self.mask]

        val_eq = torch.stack([c(state) for c in self.equality]) if self.equality else torch.tensor(0.0)
        val_ineq = []
        for j, (f, is_ge) in enumerate(self.inequality):
            val = f(state)
            if not is_ge:
                val = -val
            #pos_val = torch.clamp(val, min=-self.eta[j])
            pos_val = torch.clamp(val, min=-eta_masked)
            val_ineq.append(pos_val)

        val_ineq = torch.stack(val_ineq) if val_ineq else torch.tensor(0.0)
        return torch.norm(val_eq) + torch.norm(val_ineq)




    def isFeasible(self, state: AugLagState):
        for c in self.equality:
            if not torch.allclose(c(state), torch.tensor(0.0), atol=1e-5):
                return False
        for j, (f, is_ge) in enumerate(self.inequality):
            val = f(state)
            if not is_ge:
                val = -val
            if torch.any(val < -self.eta[j]):
                return False
        return True

    def step(self, state: AugLagState):
        assert self.isFeasible(state), "Initial point isn't feasible"

        x = self.solver.step(state.positions , eps=self.eps, max_iter=self.max_iter)
        aug_func(self, x)
        logging.info(x)
        updated_state = self._update_state(x)
        # if self.equality or self.inequality:
        #     self.update_lam(updated_state)

        count = 1
        while (self.equality or self.inequality) and self.criteria(updated_state) > self.eps:
            logger.info(f"round {count}:")
            self.sigma.data *= 10
            self.update_eta()
            count += 1
            x = self.solver.step(updated_state.positions, eps=self.eps, max_iter=self.max_iter)
            updated_state = self._update_state(x)
            aug_func(self, x)
            self.update_lam(updated_state)

        return updated_state

    class LBFGSWrapper:
        def __init__(self, func, max_iter=100, eps=1e-6):
            self.func = func
            self.max_iter = max_iter
            self.eps = eps
            self.last_loss = None  # Optional: store last computed loss

        def step(self, x, eps=None, max_iter=None, return_loss=False):
            eps = eps if eps is not None else self.eps
            max_iter = max_iter if max_iter is not None else self.max_iter

            # x = x.detach().clone().requires_grad_(True)
            optimizer = torch.optim.LBFGS([x], max_iter=max_iter, tolerance_grad=eps)

            def closure():
                optimizer.zero_grad()
                loss = self.func(x)
                logging.info(f"loss: {self.func}")
                loss.backward()
                self.last_loss = loss.detach()  # Store for inspection
                return loss

            optimizer.step(closure)

            if return_loss:
                return x, self.last_loss
            return x


# def augmented_lagrangian(x, lambda_, rho, g, f):
#     penalty = rho / 2 * torch.relu(g(x)) ** 2
#     lagrangian = f(x) + lambda_ * torch.relu(g(x)) + penalty
#     return lagrangian
# # Usage:
# x = torch.tensor(3.0, requires_grad=True)
# lambda_ = torch.tensor(1.0)
# rho = 10.0
# f = lambda x: x ** 2
# g = lambda x: x - 2
# loss = augmented_lagrangian(x, lambda_, rho, g, f)
# loss.backward()
