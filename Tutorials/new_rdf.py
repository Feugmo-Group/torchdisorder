# %%
# /// script
# dependencies = [
#     "torch",
#     "matplotlib",
#     "ase",
#     "torch_sim_atomistic[mace, io]"
# ]
# ///

import torch
import torch_sim as ts
import matplotlib.pyplot as plt
import numpy as np

from ase.build import bulk
from torch_sim.neighbors import torch_nl_linked_cell


# ============================================================
# Descriptor Class
# ============================================================

class StructureDescriptors:

    def __init__(self, cutoff=6.0, sigma_r=0.05, sigma_theta=0.05):
        self.cutoff = cutoff
        self.sigma_r = sigma_r
        self.sigma_theta = sigma_theta

    # --------------------------------------------------------
    # Neighbor list
    # --------------------------------------------------------

    def build_neighbor_list(self, state):

        from torch_sim.neighbors import torch_nl_linked_cell

        # Note: Build a neighbor list
        mapping, system_mapping, shifts = torch_nl_linked_cell(
            state.positions,
            state.cell,
            state.pbc,
            torch.tensor(self.cutoff, device=state.device),
            state.system_idx,
        )

        return mapping, shifts

    # --------------------------------------------------------
    # Pair distances
    # --------------------------------------------------------

    def pair_distances(self, state, mapping, shifts):

        rij = state.positions[mapping[0]] - state.positions[mapping[1]]

        if shifts is not None:
            rij = rij - shifts

        dist = torch.sqrt((rij ** 2).sum(dim=1) + 1e-12)

        return dist, rij

    # --------------------------------------------------------
    # Radial distribution function
    # --------------------------------------------------------

    def compute_rdf(self, state, r_bins):

        mapping, shifts = self.build_neighbor_list(state)

        r_ij, _ = self.pair_distances(state, mapping, shifts)

        sigma = self.sigma_r

        gauss = torch.exp(
            -0.5 * ((r_bins[None, :] - r_ij[:, None]) / sigma) ** 2
        ) / (sigma * np.sqrt(2 * np.pi))

        g_r = gauss.sum(dim=0)

        volume = torch.det(state.cell[0]).abs()
        density = state.n_atoms / volume

        g_r = g_r / (density * 4 * np.pi * r_bins ** 2 * state.n_atoms)

        return g_r

    # --------------------------------------------------------
    # Angular distribution function
    # --------------------------------------------------------

    def compute_adf(self, state, theta_bins):

        mapping, shifts = self.build_neighbor_list(state)

        _, rij = self.pair_distances(state, mapping, shifts)

        i = mapping[0]

        neighbors = {}

        for idx, center in enumerate(i):
            neighbors.setdefault(center.item(), []).append(idx)

        angles = []

        for center, idxs in neighbors.items():

            if len(idxs) < 2:
                continue

            vecs = rij[idxs]

            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):
                    v1 = vecs[a]
                    v2 = vecs[b]

                    cos_theta = torch.dot(v1, v2) / (
                            torch.norm(v1) * torch.norm(v2) + 1e-12
                    )

                    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

                    angles.append(theta)

        angles = torch.stack(angles)

        sigma = self.sigma_theta

        gauss = torch.exp(
            -0.5 * ((theta_bins[None, :] - angles[:, None]) / sigma) ** 2
        ) / (sigma * np.sqrt(2 * np.pi))

        adf = gauss.sum(dim=0)

        return adf


# ============================================================
# Create Structure (ASE → TorchSim)
# ============================================================

# Create a simple silicon structure using ASE
si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)

# Convert ASE Atoms to TorchSim State
state = ts.initialize_state(
    si_atoms,
    device=torch.device("cpu"),
    dtype=torch.float64
)

print("State information")
print("Atoms:", state.n_atoms)
print("Systems:", state.n_systems)


# ============================================================
# Initialize descriptor model
# ============================================================

desc = StructureDescriptors(
    cutoff=6.0,
    sigma_r=0.05,
    sigma_theta=0.05
)

# ============================================================
# RDF calculation
# ============================================================

r_bins = torch.linspace(0.1, 6.0, 200)

g_r = desc.compute_rdf(state, r_bins)


# ============================================================
# Angular distribution calculation
# ============================================================

theta_bins = torch.linspace(0, torch.pi, 180)

adf = desc.compute_adf(state, theta_bins)


# ============================================================
# Plot results
# ============================================================

plt.figure()
plt.plot(r_bins.detach().numpy(), g_r.detach().numpy())
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function")
plt.show()


plt.figure()
plt.plot(theta_bins.detach().numpy(), adf.detach().numpy())
plt.xlabel("Angle (rad)")
plt.ylabel("ADF")
plt.title("Angular Distribution Function")
plt.show()

