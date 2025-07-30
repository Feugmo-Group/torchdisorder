import torch
from torch import Tensor
from typing import Callable, Optional, Sequence, Tuple


class BarrierOptimizer:
    def __init__(
        self,
        objective: Callable[[Tensor], Tensor],
        equality: Optional[Sequence[Callable[[Tensor], Tensor]]] = None,
        inequality: Optional[Sequence[Tuple[Callable[[Tensor], Tensor], bool]]] = None,
        plot: bool = False,
    ):
        self.objective = objective
        self.equality = equality
        self.inequality = inequality
        self.plot = plot
        self.solver = None

    def is_feasible(self, x: Tensor) -> bool:
        if self.equality is not None:
            for eq in self.equality:
                if not torch.allclose(eq(x), torch.tensor(0.0, device=x.device)):
                    return False
        if self.inequality is not None:
            for ineq, is_positive in self.inequality:
                val = ineq(x)
                if is_positive:
                    if torch.any(val <= 0):
                        return False
                else:
                    if torch.any(val >= 0):
                        return False
        return True

    def set_solver(self, method: str, ls=None, sc=None):
        import optim  # assumes this is your custom optimization module
        if method == "DFP":
            self.solver = optim.DFP(self.func, ls, sc)
        elif method == "BFGS":
            self.solver = optim.BFGS(self.func, ls, sc)
        else:
            raise ValueError(f"Unknown method: {method}")


class InverseBarrier(BarrierOptimizer):
    def __init__(self, objective, mu, *constraints, plot=False):
        """
        Inverse barrier method optimizer.
        mu: barrier parameter controlling penalty magnitude
        """
        eqs, ineqs = reconstructure(constraints)
        super().__init__(objective, eqs, ineqs, plot)
        self.mu = torch.as_tensor(mu, dtype=torch.float32)

    def func(self, x):
        penalty = 0.0
        if self.equality:
            penalty += sum(eq(x)**2 for eq in self.equality)
        if self.inequality:
            for ineq, is_pos in self.inequality:
                val = ineq(x)
                penalty += 1 / val if is_pos else -1 / val
        return self.mu * penalty + self.objective(x)

    def criteria(self, x):
        total = 0.0
        if self.inequality:
            for ineq, is_pos in self.inequality:
                val = ineq(x)
                total += 1 / val if is_pos else -1 / val
        return total

    def step(self, x, eps1=1e-6, epsilon=1e-6):
        assert self.is_feasible(x), "Initial point isn't feasible"
        x = torch.as_tensor(x, dtype=torch.float32)
        count = 1
        print(f"round {count}:")
        x = self.solver.step(x, eps1)
        if self.plot:
            self.solver.plot({"inv_barrier": [self.mu]})
        while self.mu * self.criteria(x) > epsilon:
            self.mu /= 10
            count += 1
            print(f"round {count}:")
            x = self.solver.step(x, eps1)
            if self.plot:
                self.solver.plot({"inv_barrier": [self.mu]})
        return x


class LogBarrier(BarrierOptimizer):
    def __init__(self, objective, mu, *constraints, plot=False):
        """
        Log barrier method optimizer.
        mu: barrier parameter controlling penalty magnitude
        """
        eqs, ineqs = reconstructure(constraints)
        super().__init__(objective, eqs, ineqs, plot)
        self.mu = torch.as_tensor(mu, dtype=torch.float32)

    def func(self, x):
        penalty = 0.0
        if self.equality:
            penalty += sum(eq(x)**2 for eq in self.equality)
        if self.inequality:
            for ineq, is_pos in self.inequality:
                val = ineq(x)
                penalty += torch.log(val) if is_pos else torch.log(-val)
        return -self.mu * penalty + self.objective(x)

    def criteria(self, x):
        total = 0.0
        if self.inequality:
            for ineq, is_pos in self.inequality:
                val = ineq(x)
                total += torch.log(val) if is_pos else torch.log(-val)
        return total

    def step(self, x, eps1=1e-6, epsilon=1e-6):
        assert self.is_feasible(x), "Initial point isn't feasible"
        x = torch.as_tensor(x, dtype=torch.float32)
        count = 1
        print(f"round {count}:")
        x = self.solver.step(x, eps1)
        if self.plot:
            self.solver.plot({"log_barrier": [self.mu]})
        while self.criteria(x) > epsilon:
            self.mu *= 0.1
            count += 1
            print(f"round {count}:")
            x = self.solver.step(x, eps1)
            if self.plot:
                self.solver.plot({"log_barrier": [self.mu]})
        return x


class PenaltyMethod(BarrierOptimizer):
    def __init__(self, objective, sigma, *constraints, plot=False):
        """
        Penalty method optimizer.
        sigma: penalty scaling parameter
        """
        eqs, ineqs = reconstructure(constraints)
        super().__init__(objective, eqs, ineqs, plot)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32)

    def func(self, x):
        penalty = 0.0
        if self.equality:
            penalty += sum(eq(x)**2 for eq in self.equality)
        if self.inequality:
            for ineq, is_pos in self.inequality:
                val = ineq(x)
                penalty += torch.min(val, torch.tensor(0.0))**2 if is_pos else torch.min(-val, torch.tensor(0.0))**2
        return self.sigma * penalty / 2 + self.objective(x)

    def criteria(self, x):
        penalty = 0.0
        if self.equality:
            penalty += sum(eq(x)**2 for eq in self.equality)
        if self.inequality:
            for ineq, is_pos in self.inequality:
                val = ineq(x)
                penalty += torch.min(val, torch.tensor(0.0))**2 if is_pos else torch.min(-val, torch.tensor(0.0))**2
        return torch.sqrt(penalty)

    def step(self, x, eps1=1e-6, epsilon=1e-6):
        assert self.is_feasible(x), "Initial point isn't feasible"
        x = torch.as_tensor(x, dtype=torch.float32)
        count = 1
        print(f"round {count}:")
        x = self.solver.step(x, eps1)
        if self.plot:
            self.solver.plot({"penalty": [self.sigma]})
        while self.criteria(x) > epsilon:
            self.sigma *= 10
            count += 1
            print(f"round {count}:")
            x = self.solver.step(x, eps1)
            if self.plot:
                self.solver.plot({"penalty": [self.sigma]})
        return x


def reconstructure(constraints):
    eqs = []
    ineqs = []
    for c in constraints:
        if isinstance(c, tuple):
            ineqs.append(c)  # (function, sign)
        else:
            eqs.append(c)
    return eqs if eqs else None, ineqs if ineqs else None


# === q_tetrahedral constraint example wrapper ===

