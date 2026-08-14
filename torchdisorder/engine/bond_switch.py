"""Wooten-Winer-Weaire bond switching for crystal -> amorphous transformation.

Why this exists
---------------
Gradient descent on atomic positions cannot turn a crystal into a glass, and the
reason is structural rather than a matter of tuning. Measured on c-SiO2 expanded
to glass density against the published GAP glass:

    crystal seed : 375 Si,  750 Si-O-Si links  (4.00 per Si)
    published GAP: 1728 Si, 3444 Si-O-Si links (3.99 per Si)

The two have the *same* connectivity degree. They differ only in WHICH silicons
are linked -- the ring statistics. Converting one to the other means breaking and
remaking bonds. In the coordinates the optimizer can move, a single bond swap
requires an oxygen to travel ~1.94 A (from a 1.715 A first shell to the next
available site at 3.652 A), while ``stability.max_displacement`` caps each step at
0.1 A and every intermediate is uphill in both chi^2 and the constraint penalty.
Roughly twenty consecutive uphill steps, with no thermal energy to pay for them.

Wooten, Winer & Weaire (Phys. Rev. Lett. 54, 1392 (1985)) solved this with a
discrete move: transpose two bonds, relax, accept or reject. That is complementary
to what this codebase already does well -- fast differentiable local relaxation is
exactly what the inner loop of a WWW scheme needs.

Constraint handling
-------------------
Constraints are evaluated only AFTER relaxation. A bond switch passes through a
CN=3 intermediate, which the coordination constraint penalises and the overlap
floor bounds from the other side; scoring that intermediate would reject every
move before it began. Because the intermediate is never a candidate state, it
never needs scoring, and no constraint has to be weakened or special-cased.

The move
--------
For a corner-sharing tetrahedral network (Si-O-Si), pick two bridging oxygens,
O1 bridging (Si_a, Si_b) and O2 bridging (Si_c, Si_d), and rewire them to bridge
(Si_a, Si_c) and (Si_b, Si_d). Every silicon keeps its coordination number, so the
move preserves stoichiometry and CN exactly while changing the ring statistics --
which is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["BondSwitchMC", "SwitchProposal", "find_bridges"]


@dataclass
class SwitchProposal:
    """One proposed bond transposition."""

    o1: int
    o2: int
    si_a: int
    si_b: int
    si_c: int
    si_d: int
    new_o1_pos: np.ndarray
    new_o2_pos: np.ndarray

    def describe(self) -> str:
        return (f"O{self.o1}: Si{self.si_a}-Si{self.si_b} -> Si{self.si_a}-Si{self.si_c}; "
                f"O{self.o2}: Si{self.si_c}-Si{self.si_d} -> Si{self.si_b}-Si{self.si_d}")


def find_bridges(atoms, central_z: int, bridge_z: int, cutoff: float) -> Dict[int, List[int]]:
    """Map each bridging atom to the central atoms it connects.

    Only genuine bridges (exactly two central neighbours) are returned; terminal
    and over-coordinated sites are not valid endpoints for a transposition.
    """
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    i, j, _ = neighbor_list("ijd", atoms, cutoff)
    mask = (z[i] == bridge_z) & (z[j] == central_z)

    bridges: Dict[int, List[int]] = {}
    for b, c in zip(i[mask].tolist(), j[mask].tolist()):
        bridges.setdefault(b, []).append(c)
    return {b: cs for b, cs in bridges.items() if len(cs) == 2}


class BondSwitchMC:
    """Metropolis Monte Carlo over WWW bond transpositions.

    Args:
        atoms: starting structure (a crystal, typically).
        central_z / bridge_z: e.g. 14 / 8 for SiO2, 32 / 8 for GeO2.
        cutoff: first-shell cutoff for the central-bridge bond.
        relax_fn: ``f(atoms) -> atoms`` performing local relaxation. Supply the
            gradient machinery here; without it the acceptance rate collapses,
            because an unrelaxed transposition leaves badly strained bonds.
        score_fn: ``f(atoms) -> float`` returning the energy-like quantity to
            minimise (chi^2 plus constraint penalties). Lower is better.
        temperature: Metropolis temperature in units of ``score_fn``. Controls how
            much uphill motion is tolerated; too low and the run freezes into the
            crystal, too high and it accepts everything and destroys the network.
        seed: RNG seed.
    """

    def __init__(
        self,
        atoms,
        central_z: int,
        bridge_z: int,
        cutoff: float,
        relax_fn: Optional[Callable] = None,
        score_fn: Optional[Callable] = None,
        temperature: float = 0.05,
        seed: int = 0,
        max_bond_length: Optional[float] = None,
        neighbour_radius: Optional[float] = None,
        strain_tolerance: float = 1.6,
    ):
        self.atoms = atoms.copy()
        self.central_z = central_z
        self.bridge_z = bridge_z
        self.cutoff = cutoff
        self.relax_fn = relax_fn
        self.score_fn = score_fn
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.max_bond_length = max_bond_length or cutoff
        # Only bridges within this distance are candidate swap partners.  Roughly
        # the second-neighbour separation: far enough to reach a different ring,
        # close enough that the rewired bonds land inside a first shell.
        self.neighbour_radius = neighbour_radius or 3.0 * cutoff
        # How much a freshly transposed bond may exceed the first-shell cutoff
        # before the proposal is discarded. WWW moves are strained by design and
        # relaxation is what resolves them, so this is a sanity bound, not a
        # physical bond limit.
        self.strain_tolerance = strain_tolerance

        self.n_proposed = 0
        self.n_accepted = 0
        self.history: List[Dict] = []

    # -- proposal ----------------------------------------------------------
    def propose(self) -> Optional[SwitchProposal]:
        """Pick two bridges and transpose them, or return None if none is valid."""
        bridges = find_bridges(self.atoms, self.central_z, self.bridge_z, self.cutoff)
        if len(bridges) < 2:
            return None

        keys = np.array(list(bridges))
        pos = self.atoms.get_positions()

        by_centre: Dict[int, List[int]] = {}
        for b, cs in bridges.items():
            for c in cs:
                by_centre.setdefault(int(c), []).append(int(b))

        def other(bridge, centre):
            c1, c2 = bridges[bridge]
            return int(c2) if int(c1) == centre else int(c1)

        for _ in range(200):
            # The canonical WWW move works on a CONNECTED PATH of centres,
            # a - b - c - d, and rewires to a-c and b-d.  Choosing the two bridges
            # independently does not work: selecting si_c near si_a leaves si_d
            # wherever si_c's other bridge happens to point, and the second new
            # bond then spans a median 12.6 A.  Instrumented on c-SiO2, 217 of 300
            # independent draws failed on exactly that second bond and none
            # succeeded.  Along a path every pair is within two bonds, so both new
            # bridges are reachable.
            o1 = int(self.rng.choice(keys))
            si_a, si_b = (int(x) for x in bridges[o1])
            if self.rng.random() < 0.5:
                si_a, si_b = si_b, si_a

            mid = [b for b in by_centre[si_b] if b != o1]
            if not mid:
                continue
            o_mid = int(self.rng.choice(mid))
            si_c = other(o_mid, si_b)
            if si_c in (si_a, si_b):
                continue

            tail = [b for b in by_centre[si_c] if b not in (o1, o_mid)]
            if not tail:
                continue
            o2 = int(self.rng.choice(tail))
            si_d = other(o2, si_c)
            if len({si_a, si_b, si_c, si_d}) != 4:
                continue

            new1 = self._midpoint(pos[si_a], pos[si_c])
            new2 = self._midpoint(pos[si_b], pos[si_d])

            # A WWW proposal is expected to be strained; relaxation is what makes
            # it viable, which is why relax_fn is not optional in practice.  The
            # bound here only rejects the wildly impossible.
            limit = self.strain_tolerance * self.max_bond_length
            if (self._dist(new1, pos[si_a]) > limit
                    or self._dist(new1, pos[si_c]) > limit
                    or self._dist(new2, pos[si_b]) > limit
                    or self._dist(new2, pos[si_d]) > limit):
                continue

            return SwitchProposal(int(o1), int(o2), int(si_a), int(si_b),
                                  int(si_c), int(si_d), new1, new2)
        return None

    def _mic(self, dr: np.ndarray) -> np.ndarray:
        cell = np.array(self.atoms.get_cell())
        frac = dr @ np.linalg.inv(cell)
        frac -= np.round(frac)
        return frac @ cell

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(self._mic(a - b)))

    def _midpoint(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + 0.5 * self._mic(b - a)

    def apply(self, atoms, prop: SwitchProposal):
        """Return a copy of `atoms` with the transposition applied."""
        out = atoms.copy()
        pos = out.get_positions()
        pos[prop.o1] = prop.new_o1_pos
        pos[prop.o2] = prop.new_o2_pos
        out.set_positions(pos)
        return out

    # -- the loop ----------------------------------------------------------
    def step(self) -> bool:
        """Propose, relax, score, accept or reject. Returns True if accepted."""
        if self.score_fn is None:
            raise ValueError("BondSwitchMC needs a score_fn to accept or reject moves")

        prop = self.propose()
        if prop is None:
            return False

        self.n_proposed += 1
        current = self.score_fn(self.atoms)

        trial = self.apply(self.atoms, prop)
        # Relax BEFORE scoring: the raw transposition leaves strained bonds, and
        # scoring it would reject essentially every move.  The CN=3 intermediate
        # is never a candidate state and so is never scored.
        if self.relax_fn is not None:
            trial = self.relax_fn(trial)
        proposed = self.score_fn(trial)

        delta = proposed - current
        # bool() matters: numpy comparisons yield np.bool_, which fails an
        # `is True` identity check at any call site that uses one.
        accept = bool(delta <= 0 or
                      self.rng.random() < np.exp(-delta / max(self.temperature, 1e-12)))

        self.history.append({
            "proposed": float(proposed), "current": float(current),
            "delta": float(delta), "accepted": bool(accept),
            "move": prop.describe(),
        })
        if accept:
            self.atoms = trial
            self.n_accepted += 1
        return accept

    def run(self, n_steps: int, log_every: int = 25, callback=None):
        for k in range(n_steps):
            self.step()
            if log_every and (k + 1) % log_every == 0:
                rate = self.n_accepted / max(self.n_proposed, 1)
                last = self.history[-1]["current"] if self.history else float("nan")
                # flush: a 3000-switch run emits ~30 short lines, far under the 8 KB
                # block-buffer Python uses when stdout is a file, so an unflushed
                # print leaves a multi-hour batch job looking hung until it exits.
                print(f"  switch {k+1:5d}/{n_steps}  accepted {self.n_accepted:5d} "
                      f"({100*rate:5.1f}%)  score {last:.6g}", flush=True)
                if callback is not None:
                    callback(self)
        return self.atoms

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / max(self.n_proposed, 1)
