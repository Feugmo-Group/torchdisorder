import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import logging
import torch
from typing import Callable, Optional, Dict, Tuple
import torch.nn as nn
from torchdisorder.common.utils import MODELS_PROJECT_ROOT
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.rdf import SpectrumCalculator
from torchdisorder.model.generator import generate_atoms_from_config
from torchdisorder.engine.trainer import run_optimization
import numpy as np
import torch_sim as ts
from torch_sim.io import atoms_to_state, state_to_atoms
from ase.data import chemical_symbols
from ase import Atoms
from ase.io import write, Trajectory
from torch_sim.state import DeformGradMixin, SimState
from torch_sim.optimizers import fire
from torch_sim.models.mace import MaceModel, MaceUrls
from mace.calculators.foundations_models import mace_mp
from pathlib import Path
from torchdisorder.common.utils import OrderParameter
from ase import Atoms
from torchdisorder.model.xrd import XRDModel
from torchdisorder.viz.plotting import init_live_total_correlation, init_live_total_scattering, update_live_plot
from torchdisorder.viz.plotting import plot_total_correlation, plot_total_scattering
from torchdisorder.common.utils import write_trajectories
from torch_sim.state import DeformGradMixin, SimState
import plotly.graph_objects as go
from ipywidgets import interact
from torchdisorder.model.loss import ChiSquaredObjective, ConstraintChiSquared, CooperLoss
from IPython.display import display
from plotly.subplots import make_subplots
from torchdisorder.engine.optimizer import StructureFactorCMP
from torchdisorder.engine.optimizer import cooper_optimizer, perform_fire_relaxation
from torchdisorder.engine.optimizer import apply_sequential_tetrahedral_constraint
from ase.io import Trajectory
from torch_sim.io import state_to_atoms
from pathlib import Path

from torch_sim.models.mace import MaceModel
from mace.calculators.foundations_models import mace_mp
from torchdisorder.engine.optimizer import compute_mace_energy

@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "configs"), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("data.root_dir:", cfg.data.root_dir)
    #logger.info("Loaded Hydra Configuration:\n" + OmegaConf.to_yaml(cfg))

    dtype = torch.float32
    device = torch.device(cfg.accelerator)

    # Load pretrained MACE model
    mace_raw = mace_mp(model="small", return_raw_model=True)
    mace_model = MaceModel(model=mace_raw, device=device).eval()
    #
    # Relaxation parameters
    RELAX_INTERVAL = 50000  # How often to perform relaxation (every 5 steps)
    FIRE_STEPS = 50 # Max FIRE steps during relaxation

    rdf_data = TargetRDFData.from_dict(cfg.data.data, device=cfg.accelerator)
    print(cfg)
    print(cfg.data)
    print("CFG DATA DATA", cfg.data.data)
    print("Plots Dir",cfg.output.plots_dir)

    if cfg.output.save_plots:
        plot_dir = Path(cfg.output.plots_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(cfg.output.plots_dir) / "T_r_initial_plots.html"
    fig_T_r, trace_T_r = init_live_total_correlation(r_bins=rdf_data.r_bins, T_target=rdf_data.T_r_target, out_path=filename)
    filename = Path(cfg.output.plots_dir) / "S_Q_initial_plots.html"
    fig_S_Q, trace_S_Q = init_live_total_scattering(q_bins=rdf_data.q_bins, F_target=rdf_data.F_q_target, out_path=filename)
    print(cfg.output.plots_dir)




    spec_calc = SpectrumCalculator.from_config_dict(cfg.data)

    atoms = generate_atoms_from_config(cfg.structure)
    atoms_list = [atoms]
    state = atoms_to_state(atoms_list, device=cfg.accelerator, dtype=dtype)
    state.positions.requires_grad_(True)
    state.cell.requires_grad_(True)

    class StateWrapper:
        def __init__(self, original_state):
            self.__dict__.update(original_state.__dict__)
            self.system_idx = None
            self.n_systems = None

    state = StateWrapper(state)
    state.n_systems = torch.tensor(len(atoms_list))
    atoms_per_system = torch.tensor([len(a) for a in atoms_list], device=device)
    state.system_idx = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_system
    )
    state.atomic_numbers = torch.tensor(
        [chemical_symbols.index(a.symbol) for a in atoms_list[0]], dtype=torch.int64, device=device
    )
    # print("State Atomic Numbers: ", state.atomic_numbers)

    xdr_model = XRDModel(
        spectrum_calc=spec_calc,
        rdf_data=rdf_data,
        dtype=dtype,
        device=cfg.accelerator,
    )

    results = xdr_model(state)
    print("Results:", results)


    cooper_loss = CooperLoss(target_data=rdf_data, device=cfg.accelerator)

    def loss_fn(desc: dict) -> dict:
        return cooper_loss(desc)

    base_sim_state = state
    print("base_sim_state:", base_sim_state)

    target_vec = rdf_data.F_q_target
    q_bins = rdf_data.q_bins

    cooper_problem = StructureFactorCMP(
        model=xdr_model,
        base_state=base_sim_state,
        target_vec=target_vec,
        target_kind="S_Q",  #Can switch between S_Q and T_r
        q_bins=q_bins,
        loss_fn=loss_fn,
    )

    init_fn, update_fn = cooper_optimizer(
        cmp_problem=cooper_problem,
        lr=1e-4,
        max_steps=cfg.max_steps,
        optimize_cell=cfg.optimize_cell,
        verbose=True,
    )

    cmp_state = init_fn(state)
    # print("CMP state:", cmp_state)
    # print("cmp_state attributes:")
    # for attr in dir(cmp_state):
    #     val = getattr(cmp_state, attr)
    #     print(f"Attribute: {attr}, Type: {type(val)}")

    prev_loss = None  # Initialize prev_loss for first loop check


    #Defining the parameters for the sequential optimization
    SEQUENTIAL_INTERVAL = 500  # Apply every 500 steps
    SEQUENTIAL_STEPS_PER_ATOM = 100
    SEQUENTIAL_Q_THRESHOLD = 0.85
    SEQUENTIAL_LR = 0.01
    SEQUENTIAL_FREEZE = "soft"  # Options: "none", "central", "soft"

    # Updated optimization loop

    for step in range(cfg.max_steps):
        cmp_state, base_sim_state = update_fn(cmp_state, base_sim_state)

        # Trajectory saving after each step
        if cfg.output.write_trajectory:
            traj_path = Path(cfg.output.trajectory_path)
            traj_path.mkdir(parents=True, exist_ok=True)
            ase_path = traj_path / "trajectory.traj"
            xdatcar_path = traj_path / "XDATCAR"
            xyz_path = traj_path / "trajectory.xyz"  # Add XYZ path

            atoms_list_curr = state_to_atoms(base_sim_state)
            for atoms_obj in atoms_list_curr:
                write_trajectories(atoms_obj, str(ase_path), str(xdatcar_path))

                # Append to XYZ trajectory (use 'append' mode after first write)
                write(str(xyz_path), atoms_obj, format='xyz', append=(step > 0))

            print(f"Step {step}: saved structure to trajectory (including XYZ).")

        # Apply FIRE relaxation periodically (different interval than sequential)
        if step > 0 and step % RELAX_INTERVAL == 0:
            base_sim_state = perform_fire_relaxation(
                base_sim_state, mace_model, device, dtype, max_steps=FIRE_STEPS
            )
            print(f"Step {step}: performed FIRE relaxation.")

        # Apply sequential tetrahedral constraint periodically
        if step > 0 and step % SEQUENTIAL_INTERVAL == 0:
            base_sim_state = apply_sequential_tetrahedral_constraint(
                sim_state=base_sim_state,
                xrd_model=xdr_model,
                device=device,
                dtype=dtype,
                max_steps_per_atom=SEQUENTIAL_STEPS_PER_ATOM,
                q_threshold=SEQUENTIAL_Q_THRESHOLD,
                lr=SEQUENTIAL_LR,
                freeze_strategy=SEQUENTIAL_FREEZE,
                verbose=True
            )
            print(f"Step {step}: applied sequential tetrahedral constraint.")

        # Early stopping based on loss tolerance
        if step > 0 and abs(cmp_state.loss.item() - prev_loss) < cfg.tol:
            print(f"Converged at step {step}")
            break

        prev_loss = cmp_state.loss.item()

    print("Optimization completed.")

    if cfg.output.save_plots:
        # pred_T_r = cmp_state.misc.get("Y")
        # if pred_T_r is not None:
        #     pred_T_r = pred_T_r.detach().cpu().numpy()
        #     if pred_T_r.ndim > 1:
        #         pred_T_r = pred_T_r.flatten()
        #     fig_T_r.add_trace(
        #         go.Scatter(x=rdf_data.r_bins.cpu().numpy(), y=pred_T_r, mode="lines+markers", name="Predicted T(r)")
        #     )
        #     fig_T_r.write_html(str(Path(cfg.output.plots_dir) / "T_r_final_plot.html"))

        pred_S_Q = cmp_state.misc.get("Y")
        if pred_S_Q is not None:
            pred_S_Q = pred_S_Q.detach().cpu().numpy()
            if pred_S_Q.ndim > 1:
                pred_S_Q = pred_S_Q.flatten()
            fig_S_Q.add_trace(
                go.Scatter(x=rdf_data.q_bins.cpu().numpy(), y=pred_S_Q, mode="lines+markers", name="Predicted S(Q)")
            )
            fig_S_Q.write_html(str(Path(cfg.output.plots_dir) / "S_Q_final_plot.html"))
        #
        # pred_G_r = cmp_state.misc.get("Y")
        # if pred_G_r is not None:
        #     pred_G_r = pred_G_r.detach().cpu().numpy()
        #     if pred_G_r.ndim > 1:
        #         pred_G_r = pred_G_r.flatten()
        #     fig_G_r.add_trace(
        #         go.Scatter(x=rdf_data.q_bins.cpu().numpy(), y=pred_G_r, mode="lines+markers", name="Predicted G_r")
        #     )
        #     fig_S_Q.write_html(str(Path(cfg.output.plots_dir) / "G_r_final_plot.html"))




if __name__ == "__main__":
    main()

