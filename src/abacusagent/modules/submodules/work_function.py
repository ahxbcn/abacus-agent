import os
from pathlib import Path
from typing import Literal, Optional, Dict, Any

from abacustest.lib_prepare.abacus import ReadInput, WriteInput
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.modules.util.comm import run_abacus, generate_work_path, link_abacusjob, collect_metrics
from abacusagent.modules.util.cube_manipulator import read_gaussian_cube, profile1d

RY_TO_EV = 13.60569253

def plot_averaged_elecstat_pot(
    averaged_elecstat_data,
    work_path: Path,
    axis: Literal['x', 'y', 'z'] = 'z',
    plot_filename: Optional[str] = "elecstat_pot_profile.png"
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    plt.plot(averaged_elecstat_data['data'][:, 0], averaged_elecstat_data['data'][:, 1], label='Electrostatic Potential')
    plt.xlim(0, 1)
    plt.xlabel("Fractional Coordinate along " + axis)
    plt.ylabel("Electrostatic Potential (eV)")
    plot_path = os.path.join(work_path, plot_filename)
    plt.savefig(plot_path, dpi=300)

    return plot_path

def abacus_cal_work_function(
    abacus_inputs_dir: Path,
    vacuum_direction: Literal['x', 'y', 'z'] = 'z',
) -> Dict[str, Any]:
    """
    Calculate the electrostatic potential and work function using ABACUS.
    
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        vacuum_direction (Literal['x', 'y', 'z']): The direction of the vacuum.

    Returns:
        A dictionary containing:
        - elecstat_pot_work_function_work_path (Path): Path to the ABACUS job directory calculating electrostatic potential and work function.
        - elecstat_pot_file (Path): Path to the cube file containing the electrostatic potential.
        - averaged_elecstat_pot_plot (Path): Path to the plot of the averaged electrostatic potential.
        - work_function (float): The calculated work function in eV.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacus_inputs_dir,dst=work_path,copy_files=["INPUT", "STRU"], exclude_directories=True)
        input_params = ReadInput(os.path.join(work_path, 'INPUT'))
        if input_params.get('nspin', 1) not in [1, 2]:
            raise ValueError('Only non spin-polarized and collinear spin-polarized calculation are supported for calculating electrostatic potential and work function')

        input_params['calculation'] = 'scf'
        input_params['out_pot'] = 2
        WriteInput(input_params, os.path.join(work_path, 'INPUT'))

        run_abacus(work_path)

        metrics = collect_metrics(work_path, metrics_names=['normal_end', 'converge', 'efermi'])
        if metrics['normal_end'] is not True or metrics['converge'] is not True:
            raise RuntimeError('ABACUS calculation didn\'t end normally or didn\'t reached SCF convergence')

        pot_file = os.path.join(work_path, f"OUT.{input_params.get('suffix', 'ABACUS')}/ElecStaticPot.cube")
        pot = read_gaussian_cube(pot_file)

        profile_result = profile1d(pot, axis=vacuum_direction, average=True)
        profile_result['data'][:, 1] *= RY_TO_EV  # Convert from Rydberg to eV
        v_vacuum = max(profile_result['data'][:, 1])
        work_function = v_vacuum - metrics['efermi']

        # Plot the averaged electrostatic potential
        plot_path = plot_averaged_elecstat_pot(profile_result, work_path, axis=vacuum_direction)

        return {'elecstat_pot_work_function_work_path': Path(work_path).absolute(),
                'elecstat_pot_file': Path(pot_file).absolute(),
                'averaged_elecstat_pot_plot': Path(plot_path).absolute(),
                'work_function': work_function}
    except Exception as e:
        return {'message': f"Calculating electrostatic potential and work function failed: {e}"}
