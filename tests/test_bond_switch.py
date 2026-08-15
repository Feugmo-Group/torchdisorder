"""Tests for WWW bond switching.

The point of the move is to change ring topology while leaving coordination and
stoichiometry untouched -- that is precisely what gradient descent on positions
cannot do, and what rattling destroys.
"""

import numpy as np
import pytest
from ase import Atoms

from torchdisorder.engine.bond_switch import BondSwitchMC, find_bridges


def _corner_sharing_network(n=3, a=3.1, bond=1.6):
    """A simple cubic Si-O-Si corner-sharing network: every O bridges two Si."""
    si, ox = [], []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                si.append([ix * a, iy * a, iz * a])
    si = np.array(si, float)
    for p in si:                       # one bridging O along +x, +y, +z from each Si
        for d in ([a / 2, 0, 0], [0, a / 2, 0], [0, 0, a / 2]):
            ox.append(p + np.array(d))
    ox = np.array(ox, float)
    return Atoms(f"Si{len(si)}O{len(ox)}", positions=np.vstack([si, ox]),
                 cell=[n * a] * 3, pbc=True)


def test_find_bridges_returns_only_two_coordinated_bridges():
    atoms = _corner_sharing_network()
    bridges = find_bridges(atoms, central_z=14, bridge_z=8, cutoff=2.0)
    assert bridges, "expected bridging oxygens in a corner-sharing network"
    assert all(len(v) == 2 for v in bridges.values())
    # bridge indices must be oxygens
    z = atoms.get_atomic_numbers()
    assert all(z[b] == 8 for b in bridges)
    assert all(z[c] == 14 for v in bridges.values() for c in v)


def test_switch_preserves_composition_and_atom_count():
    atoms = _corner_sharing_network()
    mc = BondSwitchMC(atoms, central_z=14, bridge_z=8, cutoff=2.0,
                      score_fn=lambda a: 0.0, seed=1)
    prop = mc.propose()
    assert prop is not None
    out = mc.apply(atoms, prop)
    assert len(out) == len(atoms)
    assert (out.get_atomic_numbers() == atoms.get_atomic_numbers()).all()


def test_switch_moves_only_the_two_bridging_atoms():
    atoms = _corner_sharing_network()
    mc = BondSwitchMC(atoms, central_z=14, bridge_z=8, cutoff=2.0,
                      score_fn=lambda a: 0.0, seed=2)
    prop = mc.propose()
    out = mc.apply(atoms, prop)
    moved = np.where(np.linalg.norm(out.get_positions() - atoms.get_positions(), axis=1) > 1e-9)[0]
    assert set(moved.tolist()) <= {prop.o1, prop.o2}


def test_proposal_uses_four_distinct_centres():
    """A degenerate rewiring would create a self-bridge or be a no-op."""
    atoms = _corner_sharing_network()
    mc = BondSwitchMC(atoms, central_z=14, bridge_z=8, cutoff=2.0,
                      score_fn=lambda a: 0.0, seed=3)
    for _ in range(20):
        prop = mc.propose()
        if prop is None:
            continue
        assert len({prop.si_a, prop.si_b, prop.si_c, prop.si_d}) == 4


def _run_until_proposed(mc, limit=50):
    """step() is a no-op when no valid proposal is drawn; retry until one is.

    Not every random draw yields a physically sensible transposition, so a test
    that calls step() once can pass or fail on the RNG rather than on behaviour.
    """
    for _ in range(limit):
        accepted = mc.step()
        if mc.n_proposed:
            return accepted
    raise AssertionError("no valid proposal in %d attempts" % limit)


def test_downhill_moves_are_always_accepted():
    atoms = _corner_sharing_network()
    state = {"current": 1.0}

    def score(a):
        # first call in a step scores the current state, second the trial
        v = state["current"]
        state["current"] = 0.0 if v == 1.0 else 1.0
        return v

    mc = BondSwitchMC(atoms, central_z=14, bridge_z=8, cutoff=2.0,
                      score_fn=score, seed=4)
    assert _run_until_proposed(mc) is True
    assert mc.n_accepted == 1


def test_uphill_moves_are_rejected_at_zero_temperature():
    atoms = _corner_sharing_network()
    calls = {"n": 0}

    def score(a):
        calls["n"] += 1
        return 0.0 if calls["n"] % 2 == 1 else 10.0   # proposed always worse

    mc = BondSwitchMC(atoms, central_z=14, bridge_z=8, cutoff=2.0,
                      score_fn=score, temperature=1e-12, seed=5)
    assert _run_until_proposed(mc) is False
    assert mc.n_accepted == 0
    assert mc.n_proposed >= 1, "an uphill move must still count as proposed"


def test_both_states_get_the_same_relaxation_before_scoring():
    """Current and trial must be relaxed equally before they are compared.

    The raw transposition is strained, so the trial has to be relaxed or every
    move is rejected. But relaxing ONLY the trial is worse than not relaxing at
    all: the structural score falls monotonically under further relaxation, so
    delta then measures "extra relaxation steps" rather than "did this switch
    help", and every move is accepted. Observed on SiO2 as 3000/3000 accepted
    while rewiring 0 of 750 links -- a chain that had degenerated into plain
    relaxation while reporting perfect success.
    """
    atoms = _corner_sharing_network()
    order = []

    def relax(a):
        order.append("relax")
        return a

    def score(a):
        order.append("score")
        return 0.0

    mc = BondSwitchMC(atoms, central_z=14, bridge_z=8, cutoff=2.0,
                      relax_fn=relax, score_fn=score, seed=6)
    mc.step()
    # relax current, score it, relax trial, score it -- symmetric budgets
    assert order[:4] == ["relax", "score", "relax", "score"]


def test_step_without_score_fn_is_an_error():
    atoms = _corner_sharing_network()
    mc = BondSwitchMC(atoms, central_z=14, bridge_z=8, cutoff=2.0, seed=7)
    with pytest.raises(ValueError, match="score_fn"):
        mc.step()
