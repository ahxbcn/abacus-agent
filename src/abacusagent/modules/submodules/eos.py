import os
from pathlib import Path
from typing import Literal, List
import copy
import numpy as np
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_model.comm_eos import eos_fit
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.modules.util.comm import run_abacus, link_abacusjob, generate_work_path, collect_metrics

def is_cubic(cell: List[List[float]]) -> bool:
    """
    Check if the cell is cubic.

    Args:
        cell (List[List[float]]): The cell vectors.

    Returns:
        bool: True if the cell is cubic, False otherwise.
    """
    a, b, c = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
    len_a, len_b, len_c = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    alpha = np.arccos(np.dot(b, c) / (len_b * len_c))
    beta = np.arccos(np.dot(a, c) / (len_a * len_c))
    gamma = np.arccos(np.dot(a, b) / (len_a * len_b))
    if np.isclose(len_a, len_b) and np.isclose(len_b, len_c):
        if np.isclose(alpha, np.pi / 2) and np.isclose(beta, np.pi / 2) and np.isclose(gamma, np.pi / 2):
            return True
        else:
            return False
    else:
        return False

def plot_eos(lat_params, fit_energy, scaled_lat_params, energies):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(lat_params, fit_energy, label='Fitted Birch-Murnaghan EOS', color='red')
    plt.scatter(scaled_lat_params, energies, label='Calculated Energies', color='blue')
    plt.xlabel('Lattice Parameter (Angstrom)')
    plt.ylabel('Energy (eV)')
    plt.title('Birch-Murnaghan EOS Fit')
    plt.legend()
    plt.grid()
    plt.savefig('birch_murnaghan_eos_fit.png', dpi=300)
    fig_path = Path('birch_murnaghan_eos_fit.png').absolute()

    return fig_path

def abacus_eos(
    abacus_inputs_dir: Path,
    stru_scale_number: int = 3,
    scale_stepsize: float = 0.02
):
    """
    Use Birch-Murnaghan equation of state (EOS) to calculate the EOS data. The shape of fitted crystal is limited to cubic now.

    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        stru_scale_number (int): Number of structures to generate for EOS calculation.
        scale_stepsize (float): Step size for scaling. Default is 0.02, which means 2% of the original cell size.

    Returns:
        Dict[str, Any]: A dictionary containing EOS calculation results:
            - "eos_work_path" (Path): Working directory for the EOS calculation.
            - "eos_fig_path" (Path): Path to the EOS fitting plot (energy vs. volume).
            - "E0" (float): Minimum energy (in eV) from the EOS fit.
            - "V0" (float): Equilibrium volume (in Å³) corresponding to E0.
            - "B0" (float): Bulk modulus (in GPa) at equilibrium volume.
            - "B0_deriv" (float): Pressure derivative of the bulk modulus.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")

        work_path = Path(generate_work_path()).absolute()

        input_params = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
        input_stru_file = input_params.get('stru_file', 'STRU')
        input_stru = AbacusStru.ReadStru(os.path.join(abacus_inputs_dir, input_stru_file))

        # Generated lattice parameters for EOS calculation
        original_cell = input_stru.get_cell()
        original_cell_param = np.linalg.norm(original_cell[0])
        scales = [1 + i * scale_stepsize for i in range(-stru_scale_number, stru_scale_number + 1)]
        scaled_lat_params = [original_cell_param * scale for scale in scales]

        input_params["calculation"] = 'cell-relax'
        input_params['fixed_axes'] = 'volume'
        input_params['force_thr_ev'] = 0.01
        input_params['stress_thr'] = 1.0
        WriteInput(input_params, os.path.join(abacus_inputs_dir, "INPUT"))

        scale_cell_job_dirs = []
        stru = copy.deepcopy(input_stru)
        for i in range(len(scales)):
            dir_name = Path(os.path.join(work_path, f"scale_cell_{i}")).absolute()
            os.makedirs(dir_name, exist_ok=True)
            scale_cell_job_dirs.append(dir_name)

            link_abacusjob(
                src=abacus_inputs_dir,
                dst=Path(dir_name).absolute(),
                copy_files=["INPUT", input_stru_file],
                exclude=["OUT.*", "*.log", "*.out", "*.json", "log"],
                exclude_directories=True
            )

            new_cell = (np.array(input_stru.get_cell()) * scales[i]).tolist()
            stru.set_cell(new_cell, bohr=False, change_coord=True)
            stru.write(os.path.join(dir_name, input_stru_file))

        run_abacus(scale_cell_job_dirs)

        energies = []
        for i, job_dir in enumerate(scale_cell_job_dirs):
            metrics = collect_metrics(job_dir)
            if metrics['normal_end'] is not True or metrics['converge'] is not True:
                raise RuntimeError(f"Job {i} did not end normally or did not converge. Please check the job directory: {job_dir}")
            energies.append(metrics['energy'])

        volumes = [x**3 for x in scaled_lat_params]
        V0, E0, fit_volume, fit_energy, B0, B0_deriv, residual0 = eos_fit(volumes, energies)
        lat_params = np.cbrt(np.array(fit_volume))

        fig_path = plot_eos(lat_params, fit_energy, scaled_lat_params, energies)

        return {
            "eos_work_path": work_path.absolute(),
            "eos_fig_path": fig_path.absolute(),
            "E0": E0,
            "V0": V0,
            "B0": B0,
            "B0_deriv": B0_deriv, }
    except Exception as e:
        return {"message": f"Fitting EOS failed: {e}"}
