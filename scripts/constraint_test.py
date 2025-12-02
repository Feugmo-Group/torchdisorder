import hydra
from omegaconf import DictConfig
from ase.io import read
import torch
from torch_sim.io import atoms_to_state
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.xrd import XRDModel
from pathlib import Path
from torchdisorder.common.utils import MODELS_PROJECT_ROOT
from ase.data import chemical_symbols
from torchdisorder.model.rdf import SpectrumCalculator


@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "configs"), config_name="config", version_base="1.3")
def test_qtet(cfg: DictConfig) -> None:
    print("Running tetrahedrality q_tet test on CIF structure.")
    device = torch.device(cfg.accelerator)
    dtype = torch.float32

    spec_calc = SpectrumCalculator.from_config_dict(cfg.data)

    # Load CIF file from config path
    cif_path = Path(cfg.data.root_dir) / "crystal-structures" / "c-SiO2.cif"
    atoms = read(str(cif_path))
    print(f"Loaded CIF structure with {len(atoms)} atoms.")

    atoms_list = [atoms]
    state = atoms_to_state(atoms_list, device=device, dtype=dtype)
    state.positions.requires_grad_(True)
    state.cell.requires_grad_(True)

    class StateWrapper:
        def __init__(self, original_state):
            self.__dict__.update(original_state.__dict__)
            self.system_idx = None
            self.n_systems = None

    state = StateWrapper(state)
    state.n_systems = torch.tensor(len(atoms_list), device=device)
    atoms_per_system = torch.tensor([len(a) for a in atoms_list], device=device)
    state.system_idx = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_system
    )
    state.atomic_numbers = torch.tensor(
        [chemical_symbols.index(a.symbol) for a in atoms_list[0]], dtype=torch.int64, device=device
    )

    # Load target RDF data from config
    rdf_data = TargetRDFData.from_dict(cfg.data.data, device=device)

    # Instantiate XRD model with tetrahedrality computation enabled
    xrd_model = XRDModel(
        spectrum_calc=spec_calc,  # Not needed since only q_tet is tested; modify if required
        rdf_data=rdf_data,
        dtype=dtype,
        device=device,
        compute_q_tet=True,
    )

    # Run forward pass to compute descriptors including q_tet
    with torch.no_grad():
        results = xrd_model(state)

    q_tet = results.get("q_tet", None)
    if q_tet is not None:
        print(f"Computed tetrahedrality q_tet value: {q_tet.item()}")
    else:
        print("q_tet was not computed or returned.")

if __name__ == "__main__":
    test_qtet()
