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
from torchdisorder.engine.optimizer import StructureFactorCMPWithConstraints, perform_melt_quench
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
import json
from collections import defaultdict


@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "configs"), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ====== W&B INITIALIZATION ======
    # wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # run = wandb.init(
    #     project="torchdisorder-optimization",
    #     config=wandb_config,
    #     name=f"run_{int(time.time())}",
    #     tags=["xrd", "structure_optimization"]
    # )

    print("data.root_dir:", cfg.data.root_dir)
    # logger.info("Loaded Hydra Configuration:\n" + OmegaConf.to_yaml(cfg))

    cfg.data.data

    dtype = torch.float32
    device = torch.device(cfg.accelerator)

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
    cooper_problem = StructureFactorCMPWithConstraints(
        model=xdr_model,
        base_state=state,
        target_vec=rdf_data,
        constraints_file='/home/advaitgore/PycharmProjects/torchdisorder/data-release/json/sio2_glass_constraints.json',  # Generated by glass_generator.py
        loss_fn=loss_fn,
        target_kind='S_Q',
        device='cuda',
        penalty_rho=10.0
    )

    # ============================
    # SETUP PARAMETERS + OPTIMIZERS
    # ============================
    base_sim_state.positions.requires_grad_(True)
    primal_params = [base_sim_state.positions]
 
    if getattr(cfg, "optimize_cell", False):
        base_sim_state.cell.requires_grad_(True)
        primal_params.append(base_sim_state.cell)

    primal_optimizer = torch.optim.Adam(primal_params, lr=1e-3)

    # ===== FIX: materialize dual params and fall back to constraint_dict multipliers if needed =====
    dual_params = list(cooper_problem.dual_parameters())
    if len(dual_params) == 0 and hasattr(cooper_problem, "constraint_dict"):
        dual_params = []
        for _op_name, info in cooper_problem.constraint_dict.items():
            dual_params += list(info["constraint"].multiplier.parameters())

    if len(dual_params) == 0:
        raise RuntimeError(
            "optimizer got an empty parameter list for dual variables. "
            "No dual parameters found in cooper_problem.dual_parameters() or cooper_problem.constraint_dict."
        )

    dual_optimizer = torch.optim.SGD(
        dual_params,
        lr=1e-2,
        maximize=True
    )

    cooper_opt = cooper.optim.SimultaneousOptimizer(
        cmp=cooper_problem,
        primal_optimizers=primal_optimizer,
        dual_optimizers=dual_optimizer
    )

    # ============================
    # PLATEAU DETECTION + MELT-QUENCH
    # ============================
    def melt_quench_wrapper(current_state, mq_index):
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
        window=3000,
        melt_quench_fn=melt_quench_wrapper,
        max_melt_quench=2
    )

    PLOT_UPDATE_INTERVAL = 1

    # ====== CHECKPOINTS ======
    CHECKPOINT_INTERVAL = 10000
    checkpoint_dir = Path(cfg.output.trajectory_path).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initial_loss = None
    lr_reduced = False

    # These will be set in the loop
    cmp_state = None
    loss = None
    step = 0

    def save_checkpoint(step, loss_value=None, cmp_state=None):
        try:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"

            checkpoint_data = {
                "step": step,
                "positions": base_sim_state.positions.detach().cpu(),
                "cell": base_sim_state.cell.detach().cpu(),
                "loss": loss_value,
                "primal_optimizer_state": primal_optimizer.state_dict(),
                "dual_optimizer_state": dual_optimizer.state_dict(),
            }

            if hasattr(cooper_problem, "state_dict"):
                checkpoint_data["cooper_state"] = cooper_problem.state_dict()

            torch.save(checkpoint_data, checkpoint_path)

            atoms_checkpoint = state_to_atoms(base_sim_state)
            for _i, atoms_obj in enumerate(atoms_checkpoint):
                xyz_path = checkpoint_dir / f"structure_step_{step}.xyz"
                cif_path = checkpoint_dir / f"structure_step_{step}.cif"
                write(str(xyz_path), atoms_obj, format="xyz")
                write(str(cif_path), atoms_obj, format="cif")

                xdatcar_path = checkpoint_dir / f"XDATCAR_step_{step}"
                traj_path = checkpoint_dir / f"trajectory_step_{step}.traj"
                write_trajectories(atoms_obj, None, str(xdatcar_path))
                write_trajectories(atoms_obj, str(traj_path), None)

            if cmp_state is not None:
                pred_S_Q = cmp_state.misc.get("Y")
                if pred_S_Q is not None:
                    pred_S_Q_np = pred_S_Q.detach().cpu().numpy().reshape(-1)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rdf_data.q_bins.cpu().numpy(),
                        y=rdf_data.F_q_target.cpu().numpy(),
                        mode="lines",
                        name="Target S(Q)",
                        line=dict(dash="dash")
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

            # If W&B enabled:
            # wandb.save(str(checkpoint_path))
            # wandb.save(str(xyz_path))
            # wandb.save(str(cif_path))

            print(f"\n{'=' * 60}")
            print(f"Checkpoint saved at step {step}")
            print(f"  Location: {checkpoint_dir}")
            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"Error saving checkpoint at step {step}: {e}")
            import traceback
            traceback.print_exc()

    def save_final_results():
        try:
            final_atoms = state_to_atoms(base_sim_state)
            output_dir = Path(cfg.output.trajectory_path).parent / "final_results"
            output_dir.mkdir(parents=True, exist_ok=True)

            for _i, atoms_obj in enumerate(final_atoms):
                write(str(output_dir / "final_structure.xyz"), atoms_obj, format="xyz")
                write(str(output_dir / "final_structure.cif"), atoms_obj, format="cif")

            torch.save(
                {
                    "positions": base_sim_state.positions.detach().cpu(),
                    "cell": base_sim_state.cell.detach().cpu(),
                    "step": step,
                    "loss": loss.item() if loss is not None else None,
                },
                str(output_dir / "final_state.pt")
            )

            if cfg.output.save_plots and cmp_state is not None:
                pred_S_Q = cmp_state.misc.get("Y")
                if pred_S_Q is not None:
                    pred_S_Q = pred_S_Q.detach().cpu().numpy().reshape(-1)

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

            print(f"Results saved to {output_dir}")

        except Exception as e:
            print(f"Error saving results: {e}")

    def signal_handler(sig, frame):
        print("\n\nInterrupt received! Saving results...")
        save_final_results()
        # if W&B enabled: wandb.finish()
        print("Results saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # ============================
    # MAIN TRAINING LOOP
    # ============================
    prev_loss = None

    try:
        for step in range(int(cfg.max_steps)):
            roll_out = cooper_opt.roll(
                compute_cmp_state_kwargs={
                    "positions": base_sim_state.positions,
                    "cell": base_sim_state.cell,
                    "step": step
                }
            )

            cmp_state = roll_out.cmp_state
            loss = cmp_state.loss
            misc = cmp_state.misc

            # From your pasted CMP: misc contains these keys
            chi2_loss = misc.get("chi2_loss", loss)

            # IMPORTANT: your constraints CMP creates MULTIPLE constraints (one per OP type),
            # so do NOT take only [0]; aggregate all violations:
            all_v = []
            for cstate in cmp_state.observed_constraints.values():
                v = cstate.violation
                if v is not None:
                    all_v.append(v.reshape(-1))
            violations = torch.cat(all_v) if len(all_v) else torch.zeros(1, device=device)

            if initial_loss is not None:
                current_reduction = ((initial_loss - loss.item()) / initial_loss) * 100
                current_reduction = round(current_reduction, 1)
            else:
                current_reduction = 0.0

            # ====== PLATEAU DETECTION & MELT-QUENCH ======
            print(f"DEBUG: State size BEFORE plateau check: {base_sim_state.positions.shape}")
            base_sim_state, mq_triggered = plateau_detector.check_and_trigger(
                step=step,
                current_reduction=current_reduction,
                current_state=base_sim_state
            )
            print(f"DEBUG: State size AFTER plateau check: {base_sim_state.positions.shape}")

            if mq_triggered:
                # Restore wrapper attributes (same as your first script)
                base_sim_state = StateWrapper(base_sim_state)
                base_sim_state.n_systems = torch.tensor(len(atoms_list))

                atoms_per_system_new = torch.tensor([len(a) for a in state_to_atoms(base_sim_state)], device=device)
                base_sim_state.system_idx = torch.repeat_interleave(
                    torch.arange(len(atoms_list), device=device),
                    atoms_per_system_new
                )
                base_sim_state.atomic_numbers = torch.tensor(
                    [chemical_symbols.index(a.symbol) for a in atoms_list[0]],
                    dtype=torch.int64,
                    device=device
                )

                base_sim_state.positions.requires_grad_(True)
                base_sim_state.cell.requires_grad_(True)

                # Rebuild optimizers (because tensors changed)
                primal_params = [base_sim_state.positions]
                if getattr(cfg, "optimize_cell", False):
                    primal_params.append(base_sim_state.cell)

                primal_optimizer = torch.optim.Adam(primal_params, lr=1e-3)

                # ===== FIX: materialize dual params and fall back to constraint_dict multipliers if needed =====
                dual_params = list(cooper_problem.dual_parameters())
                if len(dual_params) == 0 and hasattr(cooper_problem, "constraint_dict"):
                    dual_params = []
                    for _op_name, info in cooper_problem.constraint_dict.items():
                        dual_params += list(info["constraint"].multiplier.parameters())

                if len(dual_params) == 0:
                    raise RuntimeError(
                        "optimizer got an empty parameter list for dual variables after melt-quench. "
                        "No dual parameters found in cooper_problem.dual_parameters() or cooper_problem.constraint_dict."
                    )

                dual_optimizer = torch.optim.SGD(
                    dual_params,
                    lr=1e-2,
                    maximize=True
                )

                cooper_opt = cooper.optim.SimultaneousOptimizer(
                    cmp=cooper_problem,
                    primal_optimizers=primal_optimizer,
                    dual_optimizers=dual_optimizer
                )

                # Update CMP base_state reference
                cooper_problem.base_state = base_sim_state

                print(f"✓ Resuming optimization from melt-quenched structure at step {step}\n")

            if initial_loss is None:
                initial_loss = loss.item()
                print(f"\n{'=' * 60}")
                print(f"Initial loss: {initial_loss:.6f}")
                print(f"Target loss for LR reduction (97% decrease): {initial_loss * 0.03:.6f}")
                print(f"{'=' * 60}\n")

            # ====== LR REDUCTION ======
            if (not lr_reduced) and (initial_loss is not None):
                current_loss = loss.item()
                loss_reduction_percent = ((initial_loss - current_loss) / initial_loss) * 100
                if loss_reduction_percent >= 97.5:
                    for pg in primal_optimizer.param_groups:
                        pg["lr"] = 1e-4
                    for pg in dual_optimizer.param_groups:
                        pg["lr"] = 1e-3
                    lr_reduced = True

                    print(f"\n{'=' * 60}")
                    print(f"LEARNING RATE REDUCED at step {step}")
                    print(f"  Initial loss: {initial_loss:.6f}")
                    print(f"  Current loss: {current_loss:.6f}")
                    print(f"  Loss reduction: {loss_reduction_percent:.2f}%")
                    print(f"  New Adam LR: 1e-4 (was 1e-3)")
                    print(f"  New SGD LR: 1e-3 (was 1e-2)")
                    print(f"{'=' * 60}\n")

            avg_violation = violations.mean().item()
            max_violation = violations.max().item()
            num_violated = (violations > 0).sum().item()

            primal_lr = primal_optimizer.param_groups[0]["lr"]
            dual_lr = dual_optimizer.param_groups[0]["lr"]

            print(
                f"Step {step}: Loss={loss.item():.6f} ({current_reduction:.1f}% reduction), "
                f"Avg violation={avg_violation:.6f}, "
                f"Max violation={max_violation:.6f}, "
                f"Violated={num_violated}/{len(violations)}, "
                f"Chi2 loss={chi2_loss.item():.6f}, "
                f"Primal LR={primal_lr:.1e}, Dual LR={dual_lr:.1e}"
            )

            # If W&B enabled:
            # wandb.log(
            #     {
            #         "loss": loss.item(),
            #         "loss_reduction_percent": current_reduction,
            #         "chi2_loss": chi2_loss.item(),
            #         "avg_violation": avg_violation,
            #         "max_violation": max_violation,
            #         "num_violated_atoms": num_violated,
            #         "primal_lr": primal_lr,
            #         "dual_lr": dual_lr,
            #     },
            #     step=step
            # )

            if (step > 0) and (step % CHECKPOINT_INTERVAL == 0):
                save_checkpoint(step, loss.item(), cmp_state)

            if step % PLOT_UPDATE_INTERVAL == 0:
                pred_sq = cmp_state.misc.get("Y")
                if pred_sq is not None:
                    pred_sq_np = pred_sq.detach().cpu().numpy().flatten()
                    # If W&B enabled:
                    # wandb.log({
                    #     "S(Q)_plot": wandb.plot.line_series(
                    #         xs=rdf_data.q_bins.cpu().numpy(),
                    #         ys=[rdf_data.F_q_target.cpu().numpy(), pred_sq_np],
                    #         keys=["Target S(Q)", "Predicted S(Q)"],
                    #         title="Structure Factor",
                    #         xname="Q (Å⁻¹)"
                    #     )
                    # }, step=step)

            if cfg.output.write_trajectory:
                traj_path = Path(cfg.output.trajectory_path)
                traj_path.mkdir(parents=True, exist_ok=True)

                ase_path = traj_path / "trajectory.traj"
                xdatcar_path = traj_path / "XDATCAR"
                xyz_path = traj_path / "trajectory.xyz"

                atoms_list_curr = state_to_atoms(base_sim_state)
                for atoms_obj in atoms_list_curr:
                    write_trajectories(atoms_obj, str(ase_path), str(xdatcar_path))
                    write(str(xyz_path), atoms_obj, format="xyz", append=(step > 0))

            prev_loss = loss.item()

    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected!")
        save_final_results()
        # if W&B enabled: wandb.finish()
        print("Results saved. Exiting.")
        sys.exit(0)

    print("Optimization completed.")
    save_final_results()

    # final plot file (optional)
    if cfg.output.save_plots and (fig_S_Q is not None) and (cmp_state is not None):
        pred_S_Q = cmp_state.misc.get("Y")
        if pred_S_Q is not None:
            pred_S_Q = pred_S_Q.detach().cpu().numpy().reshape(-1)
            fig_S_Q.add_trace(go.Scatter(
                x=rdf_data.q_bins.cpu().numpy(),
                y=pred_S_Q,
                mode="lines+markers",
                name="Predicted S(Q)"
            ))
            fig_S_Q.write_html(str(Path(cfg.output.plots_dir) / "S_Q_final_plot.html"))

    # if W&B enabled:
    # wandb.finish()


if __name__ == "__main__":
    main()
