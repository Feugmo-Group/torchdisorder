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
from torchdisorder.engine.optimizer import StructureFactorCMP, perform_melt_quench
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
from torchdisorder.engine.callbacks import PlateauDetector


@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "configs"), config_name="config_4", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ====== W&B INITIALIZATION ======
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project="torchdisorder-optimization",
        config=wandb_config,
        name=f"run_{int(time.time())}",
        tags=["xrd", "structure_optimization"]
    )

    print("data.root_dir:", cfg.data.root_dir)
    # logger.info("Loaded Hydra Configuration:\n" + OmegaConf.to_yaml(cfg))

    cfg.data.data

    dtype = torch.float32
    device = torch.device(cfg.accelerator)

    mace = mace_mp(
        model="small",
        return_raw_model=True,
        default_dtype="float32"  # ← Add this to use float32
    )
    mace_model = MaceModel(model=mace, device=device, dtype=torch.float32)  # initializing mace model

    rdf_data = TargetRDFData.from_dict(cfg.data.data, device=cfg.accelerator)
    print(rdf_data)

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

    # #Running initial melt quench to amorphize structure
    #
    # print("\n" + "=" * 70)
    # print("GENERATING AMORPHOUS INITIAL STRUCTURE")
    # print("=" * 70)
    #
    # # Perform initial melt-quench to create truly amorphous structure
    # amorphous_state = perform_melt_quench(
    #     sim_state=atoms_to_state(atoms_list, device=cfg.accelerator, dtype=dtype),
    #     mace_model=mace_model,
    #     device=cfg.accelerator,
    #     dtype=dtype,
    #     melt_temp=4000,  # Higher temp to fully destroy crystalline order
    #     quench_temp=300,
    #     melt_steps=20000,  # Longer melting phase
    #     quench_steps=40000,  # Much slower quench
    #     timestep=0.5,
    #     save_prefix="initial_amorphization"
    # )
    #
    # print("\n" + "=" * 70)
    # print("AMORPHOUS STRUCTURE GENERATED - Starting optimization from this structure")
    # print("=" * 70 + "\n")
    #
    # # state = atoms_to_state(atoms_list, device=cfg.accelerator, dtype=dtype)
    # state = amorphous_state
    # state.positions.requires_grad_(True)
    # state.cell.requires_grad_(True)
    #
    #
    # class StateWrapper:
    #     def __init__(self, original_state):
    #         self.__dict__.update(original_state.__dict__)
    #         self.system_idx = None
    #         self.n_systems = None
    #
    # state = StateWrapper(state)
    # state.n_systems = torch.tensor(len(atoms_list))
    #
    # # Get atoms from the amorphous state, not original atoms_list
    # amorphous_atoms = state_to_atoms(amorphous_state)
    # atoms_per_system = torch.tensor([len(a) for a in amorphous_atoms], device=device)
    #
    # state.system_idx = torch.repeat_interleave(
    #     torch.arange(len(amorphous_atoms), device=device), atoms_per_system
    # )
    # state.atomic_numbers = torch.tensor(
    #     [chemical_symbols.index(a.symbol) for a in amorphous_atoms[0]],
    #     dtype=torch.int64, device=device
    # )

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
        q_threshold=0.7,  # SiO2
        device=cfg.accelerator,
        penalty_rho=10,  # SiO2
    )

    # Setup parameters
    base_sim_state.positions.requires_grad_(True)
    primal_params = [base_sim_state.positions]

    if cfg.optimize_cell:
        base_sim_state.cell.requires_grad_(True)
        primal_params.append(base_sim_state.cell)

    # Create optimizers and adjust lr for training here
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

    # ====== PLATEAU DETECTION SETUP ======
    def melt_quench_wrapper(current_state, mq_index):
        """Wrapper to call melt-quench with proper arguments"""
        return perform_melt_quench(
            sim_state=current_state,
            mace_model=mace_model,
            device=device,
            dtype=dtype,
            melt_temp=1200,
            quench_temp=300,
            melt_steps=2000,
            quench_steps=5000,
            timestep=1.0,
            save_prefix=f"melt_quench_{mq_index}"
        )

    plateau_detector = PlateauDetector(
        window=1000,  # Number of steps to check for plateau
        melt_quench_fn=melt_quench_wrapper,
        max_melt_quench=2  # Change this to toggle melt_quench
    )

    # monitor = LivePlotMonitor(
    #     q_bins_np=q_bins.cpu().numpy(),
    #     target_sq_np=rdf_data.F_q_target.cpu().numpy(),
    #     port=8050
    # )
    # monitor.start_server()
    # time.sleep(2)  # Give server time to start
    PLOT_UPDATE_INTERVAL = 1

    # ====== CHECKPOINT CONFIGURATION ======
    CHECKPOINT_INTERVAL = 10000  # Save every 10,000 steps
    checkpoint_dir = Path(cfg.output.trajectory_path).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ====== ADAPTIVE LEARNING RATE CONFIGURATION ======
    initial_loss = None  # Will be set on first step
    lr_reduced = False

    def save_checkpoint(step, loss_value=None, cmp_state=None):
        """Save a comprehensive checkpoint at the given step"""
        try:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"

            # Save PyTorch checkpoint
            checkpoint_data = {
                'step': step,
                'positions': base_sim_state.positions.cpu().detach(),
                'cell': base_sim_state.cell.cpu().detach(),
                'loss': loss_value,
                'primal_optimizer_state': primal_optimizer.state_dict(),
                'dual_optimizer_state': dual_optimizer.state_dict(),
            }

            if hasattr(cooper_problem, 'state_dict'):
                checkpoint_data['cooper_state'] = cooper_problem.state_dict()

            torch.save(checkpoint_data, checkpoint_path)

            # Save structure in multiple formats
            atoms_checkpoint = state_to_atoms(base_sim_state)
            for i, atoms_obj in enumerate(atoms_checkpoint):
                # XYZ format
                xyz_path = checkpoint_dir / f"structure_step_{step}.xyz"
                write(str(xyz_path), atoms_obj, format='xyz')

                # CIF format
                cif_path = checkpoint_dir / f"structure_step_{step}.cif"
                write(str(cif_path), atoms_obj, format='cif')

                # XDATCAR format
                xdatcar_path = checkpoint_dir / f"XDATCAR_step_{step}"
                write_trajectories(atoms_obj, None, str(xdatcar_path))

                # ASE trajectory format
                traj_path = checkpoint_dir / f"trajectory_step_{step}.traj"
                write_trajectories(atoms_obj, str(traj_path), None)

            # Save S(Q) plot if available
            if cmp_state is not None:
                pred_S_Q = cmp_state.misc.get("Y")
                if pred_S_Q is not None:
                    pred_S_Q_np = pred_S_Q.detach().cpu().numpy()
                    if pred_S_Q_np.ndim > 1:
                        pred_S_Q_np = pred_S_Q_np.flatten()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rdf_data.q_bins.cpu().numpy(),
                        y=rdf_data.F_q_target.cpu().numpy(),
                        mode="lines",
                        name="Target S(Q)",
                        line=dict(dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=rdf_data.q_bins.cpu().numpy(),
                        y=pred_S_Q_np,
                        mode="lines",
                        name="Predicted S(Q)"
                    ))
                    fig.update_layout(
                        title=f"Structure Factor at Step {step}",
                        xaxis_title="Q (Å⁻¹)",
                        yaxis_title="S(Q)"
                    )
                    fig.write_html(str(checkpoint_dir / f"S_Q_step_{step}.html"))

            # ====== W&B: Save checkpoint files ======
            wandb.save(str(checkpoint_path))
            wandb.save(str(xyz_path))
            wandb.save(str(cif_path))

            print(f"\n{'=' * 60}")
            print(f"Checkpoint saved at step {step}")
            print(f"  Location: {checkpoint_dir}")
            print(f"  Files saved:")
            print(f"    - checkpoint_step_{step}.pt")
            print(f"    - structure_step_{step}.xyz")
            print(f"    - structure_step_{step}.cif")
            print(f"    - XDATCAR_step_{step}")
            print(f"    - trajectory_step_{step}.traj")
            if cmp_state is not None:
                print(f"    - S_Q_step_{step}.html")
            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"Error saving checkpoint at step {step}: {e}")
            import traceback
            traceback.print_exc()

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
            # if cfg.output.save_plots and cmp_state is not None:
            #     pred_S_Q = cmp_state.misc.get("Y")
            #     if pred_S_Q is not None:
            #         pred_S_Q = pred_S_Q.detach().cpu().numpy()
            #         if pred_S_Q.ndim > 1:
            #             pred_S_Q = pred_S_Q.flatten()
            #
            #         fig = go.Figure()
            #         fig.add_trace(go.Scatter(
            #             x=rdf_data.q_bins.cpu().numpy(),
            #             y=rdf_data.F_q_target.cpu().numpy(),
            #             mode="lines",
            #             name="Target G(r)"
            #         ))
            #         fig.add_trace(go.Scatter(
            #             x=rdf_data.q_bins.cpu().numpy(),
            #             y=pred_S_Q,
            #             mode="lines+markers",
            #             name="Predicted G(r)"
            #         ))
            #         fig.write_html(str(output_dir / "G_r_final.html"))
        except Exception as e:
            print(f"Error saving results: {e}")

    # Setup signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print('\n\nInterrupt received! Saving results...')
        save_final_results()
        wandb.finish()  # ====== W&B: Finish run ======
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

            # Calculate current reduction percentage BEFORE plateau check
            if initial_loss is not None:
                current_reduction = ((initial_loss - loss.item()) / initial_loss) * 100
                current_reduction = round(current_reduction, 1)
            else:
                current_reduction = 0.0

            # ====== PLATEAU DETECTION & MELT-QUENCH (CORRECTED) ======
            base_sim_state, mq_triggered = plateau_detector.check_and_trigger(
                step=step,
                current_reduction=current_reduction,
                current_state=base_sim_state
            )

            if mq_triggered:
                # Restore critical StateWrapper attributes
                base_sim_state = StateWrapper(base_sim_state)
                base_sim_state.n_systems = torch.tensor(len(atoms_list))
                atoms_per_system_new = torch.tensor([len(a) for a in state_to_atoms(base_sim_state)], device=device)
                base_sim_state.system_idx = torch.repeat_interleave(
                    torch.arange(len(atoms_list), device=device), atoms_per_system_new
                )
                base_sim_state.atomic_numbers = torch.tensor(
                    [chemical_symbols.index(a.symbol) for a in atoms_list[0]],
                    dtype=torch.int64, device=device
                )

                # Update gradients
                base_sim_state.positions.requires_grad_(True)
                base_sim_state.cell.requires_grad_(True)

                # Reset optimizer parameters with new state
                primal_params = [base_sim_state.positions]
                if cfg.optimize_cell:
                    primal_params.append(base_sim_state.cell)

                # Recreate optimizers with new parameters
                primal_optimizer = torch.optim.Adam(primal_params, lr=1e-3)
                dual_optimizer = torch.optim.SGD(
                    cooper_problem.dual_parameters(),
                    lr=1e-2,
                    maximize=True
                )

                # Recreate Cooper optimizer
                cooper_opt = cooper.optim.SimultaneousOptimizer(
                    cmp=cooper_problem,
                    primal_optimizers=primal_optimizer,
                    dual_optimizers=dual_optimizer
                )

                # Update base_state reference in Cooper problem
                cooper_problem.base_state = base_sim_state

                print(f"✓ Resuming optimization from melt-quenched structure at step {step}\n")

            # Track initial loss
            if initial_loss is None:
                initial_loss = loss.item()
                print(f"\n{'=' * 60}")
                print(f"Initial loss: {initial_loss:.6f}")
                print(f"Target loss for LR reduction (97% decrease): {initial_loss * 0.03:.6f}")
                print(f"{'=' * 60}\n")

            # ====== ADAPTIVE LEARNING RATE REDUCTION ======
            if not lr_reduced and initial_loss is not None:
                current_loss = loss.item()
                loss_reduction_percent = ((initial_loss - current_loss) / initial_loss) * 100

                # Check if loss has decreased by 97% or more
                if loss_reduction_percent >= 97.5:
                    # Reduce learning rates
                    for param_group in primal_optimizer.param_groups:
                        param_group['lr'] = 1e-4

                    for param_group in dual_optimizer.param_groups:
                        param_group['lr'] = 1e-3

                    lr_reduced = True

                    print(f"\n{'=' * 60}")
                    print(f"LEARNING RATE REDUCED at step {step}")
                    print(f"  Initial loss: {initial_loss:.6f}")
                    print(f"  Current loss: {current_loss:.6f}")
                    print(f"  Loss reduction: {loss_reduction_percent:.2f}%")
                    print(f"  New Adam LR: 1e-4 (was 1e-3)")
                    print(f"  New SGD LR: 1e-3 (was 1e-2)")
                    print(f"{'=' * 60}\n")

            # Logging every step
            avg_violation = violations.mean().item()
            max_violation = violations.max().item()
            num_violated = (violations > 0).sum().item()

            # Get current learning rates
            primal_lr = primal_optimizer.param_groups[0]['lr']
            dual_lr = dual_optimizer.param_groups[0]['lr']

            print(f"Step {step}: Loss={loss.item():.6f} ({current_reduction:.1f}% reduction), "
                  f"Avg q_tet violation={avg_violation:.6f}, "
                  f"Max violation={max_violation:.6f}, "
                  f"Atoms violated={num_violated}/{len(violations)}, "
                  f"Chi2 loss={chi2_loss.item():.6f}, "
                  f"Primal LR={primal_lr:.1e}, Dual LR={dual_lr:.1e}")

            # ====== W&B: Log metrics ======
            wandb.log({
                "loss": loss.item(),
                "loss_reduction_percent": current_reduction,
                "chi2_loss": chi2_loss.item(),
                "avg_violation": avg_violation,
                "max_violation": max_violation,
                "num_violated_atoms": num_violated,
                "primal_lr": primal_lr,
                "dual_lr": dual_lr,
            }, step=step)

            # ====== CHECKPOINT SAVING ======
            if (step > 0) and (step % CHECKPOINT_INTERVAL == 0):
                save_checkpoint(step, loss.item(), cmp_state)

            # live dashboard update
            if step % PLOT_UPDATE_INTERVAL == 0:
                pred_sq = cmp_state.misc.get("Y")
                if pred_sq is not None:
                    pred_sq_np = pred_sq.detach().cpu().numpy().flatten()
                    # monitor.update_data(
                    #     step=step,
                    #     loss=loss.item(),
                    #     pred_sq=pred_sq_np,
                    #     num_violated=num_violated
                    # )

                    # ====== W&B: Log S(Q) plot ======
                    wandb.log({
                        "S(Q)_plot": wandb.plot.line_series(
                            xs=rdf_data.q_bins.cpu().numpy(),
                            ys=[rdf_data.F_q_target.cpu().numpy(), pred_sq_np],
                            keys=["Target S(Q)", "Predicted S(Q)"],
                            title="Structure Factor",
                            xname="Q (Å⁻¹)"

                        )
                    }, step=step)
                    # # ====== W&B: Log G(r) plot ======
                    # wandb.log({
                    #     "G(r)_plot": wandb.plot.line_series(
                    #         xs=rdf_data.q_bins.cpu().numpy(),
                    #         ys=[rdf_data.F_q_target.cpu().numpy(), pred_sq_np],
                    #         keys=["Target G(r)", "Predicted G(r)"],
                    #         title="Radial Distribution Function",
                    #         xname="r"
                    #     )
                    # }, step=step)


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
        wandb.finish()  # ====== W&B: Finish run ======
        print("Results saved. Exiting.")
        sys.exit(0)

    print("Optimization completed.")

    # Save results on normal completion
    save_final_results()

    # Final structure
    final_atoms = state_to_atoms(base_sim_state)

    if cfg.output.save_plots:
        pred_S_Q = cmp_state.misc.get("Y")
        if pred_S_Q is not None:
            pred_S_Q = pred_S_Q.detach().cpu().numpy()
            if pred_S_Q.ndim > 1:
                pred_S_Q = pred_S_Q.flatten()
            fig_S_Q.add_trace(
                go.Scatter(x=rdf_data.q_bins.cpu().numpy(), y=pred_S_Q, mode="lines+markers", name="Predicted S(Q)")
            )
            fig_S_Q.write_html(str(Path(cfg.output.plots_dir) / "S_Q_final_plot.html"))
        # pred_S_Q = cmp_state.misc.get("Y")
        # if pred_S_Q is not None:
        #     pred_S_Q = pred_S_Q.detach().cpu().numpy()
        #     if pred_S_Q.ndim > 1:
        #         pred_S_Q = pred_S_Q.flatten()
        #     fig_S_Q.add_trace(
        #         go.Scatter(x=rdf_data.q_bins.cpu().numpy(), y=pred_S_Q, mode="lines+markers", name="Predicted G(r)")
        #     )
        #     fig_S_Q.write_html(str(Path(cfg.output.plots_dir) / "G_r_final_plot.html"))

    # ====== W&B: Finish run ======
    wandb.finish()


if __name__ == "__main__":
    main()
