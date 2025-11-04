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
from torchdisorder.viz.plotting import plot_total_correlation, plot_total_scattering, LivePlotMonitor
from torchdisorder.common.utils import write_trajectories
from torch_sim.state import DeformGradMixin, SimState
import plotly.graph_objects as go
from ipywidgets import interact
from torchdisorder.model.loss import ChiSquaredObjective, ConstraintChiSquared, CooperLoss
from IPython.display import display
from plotly.subplots import make_subplots
from torchdisorder.engine.optimizer import StructureFactorCMP
from torchdisorder.engine.optimizer import cooper_optimizer
from ase.io import Trajectory
from torch_sim.io import state_to_atoms
from pathlib import Path

from torch_sim.models.mace import MaceModel, MaceUrls
from mace.calculators.foundations_models import mace_mp
import cooper
from cooper.optim import AlternatingPrimalDualOptimizer
import time
import signal
import sys


@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "configs"), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("data.root_dir:", cfg.data.root_dir)
    # logger.info("Loaded Hydra Configuration:\n" + OmegaConf.to_yaml(cfg))

    dtype = torch.float32
    device = torch.device(cfg.accelerator)

    mace = mace_mp(
        model="small",
        return_raw_model=True,
        default_dtype="float32"  # ← Add this to use float32
    )
    mace_model = MaceModel(model=mace, device=device, dtype=torch.float32)

    rdf_data = TargetRDFData.from_dict(cfg.data.data, device=cfg.accelerator)
    print(cfg)
    print("CFG.data")
    print(cfg.data)
    print("CFG DATA DATA", cfg.data.data)
    print("Plots Dir", cfg.output.plots_dir)

    if cfg.output.save_plots:
        plot_dir = Path(cfg.output.plots_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(cfg.output.plots_dir) / "T_r_initial_plots.html"
    fig_T_r, trace_T_r = init_live_total_correlation(r_bins=rdf_data.r_bins.cpu(), T_target=rdf_data.T_r_target.cpu(),
                                                     out_path=filename)
    filename = Path(cfg.output.plots_dir) / "S_Q_initial_plots.html"
    fig_S_Q, trace_S_Q = init_live_total_scattering(q_bins=rdf_data.q_bins.cpu(), F_target=rdf_data.F_q_target.cpu(),
                                                    out_path=filename)
    print(cfg.output.plots_dir)

    spec_calc = SpectrumCalculator.from_config_dict(cfg.data)

    atoms = generate_atoms_from_config(cfg.structure)
    atoms_list = [atoms]
    print("Atoms: ", atoms_list)
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

    # Right before calling xrd_model
    print(f"\nQ bins debug:")
    print(f"  Min Q: {rdf_data.q_bins.min()}")
    print(f"  Max Q: {rdf_data.q_bins.max()}")
    print(f"  First 5 Q: {rdf_data.q_bins[:5]}")
    print(f"  Has Q=0: {(rdf_data.q_bins == 0).any()}")

    xdr_model = XRDModel(
        spectrum_calc=spec_calc,
        rdf_data=rdf_data,
        dtype=dtype,
        device=cfg.accelerator,
    )

    try:
        results = xdr_model(state)
        print(results)
        print("XRD model completed successfully!")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    cooper_loss = CooperLoss(target_data=rdf_data, device=cfg.accelerator)

    def loss_fn(desc: dict) -> dict:
        return cooper_loss(desc)

    base_sim_state = state
    print("base_sim_state:", base_sim_state)

    q_bins = rdf_data.q_bins

    # Create Cooper problem
    cooper_problem = StructureFactorCMP(
        model=xdr_model,
        base_state=base_sim_state,
        target_vec=rdf_data,
        target_kind="S_Q",
        q_bins=q_bins,
        loss_fn=loss_fn,
        q_threshold=0.8,
        device=cfg.accelerator,
        penalty_rho=10,
        # mace_model=mace_model,  # Pass MACE model
        # energy_weight=50,
    )

    # Setup parameters
    base_sim_state.positions.requires_grad_(True)
    primal_params = [base_sim_state.positions]

    if cfg.optimize_cell:
        base_sim_state.cell.requires_grad_(True)
        primal_params.append(base_sim_state.cell)

    # CREATE OPTIMIZERS
    primal_optimizer = torch.optim.Adam(primal_params, lr=1e-3)
    dual_optimizer = torch.optim.SGD(
        cooper_problem.dual_parameters(),
        lr=1e-2,
        maximize=True
    )

    # CREATE COOPER OPTIMIZER
    cooper_opt = cooper.optim.SimultaneousOptimizer(
        cmp=cooper_problem,
        primal_optimizers=primal_optimizer,
        dual_optimizers=dual_optimizer
    )

    monitor = LivePlotMonitor(
        q_bins_np=q_bins.cpu().numpy(),
        target_sq_np=rdf_data.F_q_target.cpu().numpy(),
        port=8050
    )
    monitor.start_server()
    time.sleep(2)  # Give server time to start
    PLOT_UPDATE_INTERVAL = 1

    # Save function
    def save_final_results():
        """Save final structure and plots"""
        try:
            final_atoms = state_to_atoms(base_sim_state)
            output_dir = Path(cfg.output.trajectory_path).parent / "final_results"
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, atoms_obj in enumerate(final_atoms):
                write(str(output_dir / "final_structure.xyz"), atoms_obj, format='xyz')
                write(str(output_dir / "final_structure.cif"), atoms_obj, format='cif')

            torch.save({
                'positions': base_sim_state.positions.cpu(),
                'cell': base_sim_state.cell.cpu(),
                'step': step,
                'loss': loss.item() if 'loss' in locals() and loss is not None else None,
            }, str(output_dir / "final_state.pt"))

            print(f"\n{'=' * 60}")
            print(f"Results saved to {output_dir}")
            print(f"{'=' * 60}\n")

            if cfg.output.save_plots and cmp_state is not None:
                pred_S_Q = cmp_state.misc.get("Y")
                if pred_S_Q is not None:
                    pred_S_Q = pred_S_Q.detach().cpu().numpy()
                    if pred_S_Q.ndim > 1:
                        pred_S_Q = pred_S_Q.flatten()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rdf_data.q_bins.cpu().numpy(),
                        y=rdf_data.F_q_target.cpu().numpy(),
                        mode="lines",
                        name="Target S(Q)"
                    ))
                    fig.add_trace(go.Scatter(
                        x=rdf_data.q_bins.cpu().numpy(),
                        y=pred_S_Q,
                        mode="lines+markers",
                        name="Predicted S(Q)"
                    ))
                    fig.write_html(str(output_dir / "S_Q_final.html"))
        except Exception as e:
            print(f"Error saving results: {e}")

    # Setup signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print('\n\nInterrupt received! Saving results...')
        save_final_results()
        print('Results saved. Exiting.')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # MAIN TRAINING LOOP
    prev_loss = None
    step = 0
    cmp_state = None
    loss = None

    try:
        for step in range(cfg.max_steps):
            # COOPER OPTIMIZATION STEP
            roll_out = cooper_opt.roll(
                compute_cmp_state_kwargs={
                    "positions": base_sim_state.positions,
                    "cell": base_sim_state.cell,
                    "step": step
                }
            )

            # Extract results
            cmp_state = roll_out.cmp_state
            loss = cmp_state.loss
            misc = cmp_state.misc
            chi2_loss = misc['chi2_loss']
            violations = list(cmp_state.observed_constraints.values())[0].violation

            # Logging every step
            avg_violation = violations.mean().item()
            max_violation = violations.max().item()
            num_violated = (violations > 0).sum().item()
            print(f"Step {step}: Loss={loss.item():.6f}, "
                  f"Avg q_tet violation={avg_violation:.6f}, "
                  f"Max violation={max_violation:.6f}, "
                  f"Atoms violated={num_violated}/{len(violations)},"
                  f"Chi2 loss={chi2_loss.item():.6f}")

            # live dashboard update
            if step % PLOT_UPDATE_INTERVAL == 0:
                pred_sq = cmp_state.misc.get("Y")
                if pred_sq is not None:
                    pred_sq_np = pred_sq.detach().cpu().numpy().flatten()
                    monitor.update_data(
                        step=step,
                        loss=loss.item(),
                        pred_sq=pred_sq_np,
                        num_violated=num_violated
                    )

            # Trajectory saving
            if cfg.output.write_trajectory:
                traj_path = Path(cfg.output.trajectory_path)
                traj_path.mkdir(parents=True, exist_ok=True)
                ase_path = traj_path / "trajectory.traj"
                xdatcar_path = traj_path / "XDATCAR"
                xyz_path = traj_path / "trajectory.xyz"

                atoms_list_curr = state_to_atoms(base_sim_state)
                for atoms_obj in atoms_list_curr:
                    write_trajectories(atoms_obj, str(ase_path), str(xdatcar_path))
                    write(str(xyz_path), atoms_obj, format='xyz', append=(step > 0))

            prev_loss = loss.item()

    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected!")
        save_final_results()
        print("Results saved. Exiting.")
        sys.exit(0)

    print("Optimization completed.")

    # Save results on normal completion
    save_final_results()

    # Final structure
    final_atoms = state_to_atoms(base_sim_state)

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