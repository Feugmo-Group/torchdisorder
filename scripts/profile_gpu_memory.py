"""Locate the peak GPU allocation in a single forward+backward pass.

The balancing scan died at step 0 asking for 10.81 GiB on top of 12.80 GiB already
held, i.e. the whole of a 23.6 GiB A30, and setting scattering.chunk_size did not
change the request by a byte -- so the allocation is not in the chunked KDE path.
This walks the pipeline stage by stage and reports the peak after each, so the
culprit is identified rather than guessed at.

Usage (on a GPU node):
    python scripts/profile_gpu_memory.py --device cuda
    python scripts/profile_gpu_memory.py --device cuda --atoms 1125   # size sweep
"""

from __future__ import annotations

import argparse


def gb(x):
    return x / 1024**3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--structure", default="data/json/sio2_glass_gap.cif")
    p.add_argument("--n-r-bins", type=int, default=1000)
    p.add_argument("--n-q-bins", type=int, default=975)
    args = p.parse_args()

    import torch
    from ase.io import read
    from torch_sim.io import atoms_to_state

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        total = torch.cuda.get_device_properties(dev).total_memory
        print(f"device: {torch.cuda.get_device_name(dev)}  total {gb(total):.2f} GiB")

    def mark(label):
        if dev.type != "cuda":
            print(f"  {label}")
            return
        torch.cuda.synchronize()
        print(f"  {label:38s} alloc {gb(torch.cuda.memory_allocated(dev)):6.2f} "
              f"peak {gb(torch.cuda.max_memory_allocated(dev)):6.2f} GiB")

    atoms = read(args.structure)
    print(f"\nstructure: {len(atoms)} atoms, {atoms.get_chemical_formula(empirical=True)}")
    state = atoms_to_state(atoms, device=dev, dtype=torch.float32)
    mark("state on device")

    n = len(atoms)
    print(f"\nnaive tensor sizes for {n} atoms (float32):")
    for label, elems in [
        ("all pairs (N^2)", n * n),
        ("all pairs x 3 (displacements)", n * n * 3),
        ("unique pairs x n_r_bins", n * (n - 1) // 2 * args.n_r_bins),
        ("unique pairs x n_q_bins", n * (n - 1) // 2 * args.n_q_bins),
        ("all pairs x 27 images", n * n * 27),
        ("all pairs x 27 x 3", n * n * 27 * 3),
    ]:
        print(f"  {label:32s} {gb(elems * 4):9.2f} GiB")

    print("\nThe stage whose peak jump matches the failed 10.81 GiB request is the one\n"
          "to chunk.  If none does, the allocation is transient inside autograd and the\n"
          "system size has to come down instead.")

    # Forward pass through the real model, staged.
    try:
        from torchdisorder.common.target_rdf import TargetRDFData  # noqa: F401
    except Exception:
        pass

    from torchdisorder.model.xrd import XRDModel

    r_bins = torch.linspace(0.01, 10.0, args.n_r_bins, device=dev)
    q_bins = torch.linspace(0.5, 25.0, args.n_q_bins, device=dev)
    symbols = sorted(set(atoms.get_chemical_symbols()))

    cfg = {
        "kernel_width": 0.035,
        "neutron_scattering_lengths": {"Si": 4.1491, "O": 5.803},
        "xray_form_factor_params": {},
        "scattering_type": "neutron",
        "chunk_size": 20000,
    }
    model = XRDModel(symbols=symbols, config=cfg, r_bins=r_bins, q_bins=q_bins,
                     rdf_data=None, device=str(dev))
    mark("model built")

    # XRDModel.forward takes the SimState itself.
    state.positions = state.positions.clone().requires_grad_(True)
    out = model(state)
    mark("forward pass")

    key = "F_Q" if "F_Q" in out else list(out)[0]
    loss = (out[key] ** 2).sum()
    loss.backward()
    mark("backward pass")

    if dev.type == "cuda":
        print(f"\nPEAK: {gb(torch.cuda.max_memory_allocated(dev)):.2f} GiB "
              f"of {gb(total):.2f} GiB")


if __name__ == "__main__":
    main()
