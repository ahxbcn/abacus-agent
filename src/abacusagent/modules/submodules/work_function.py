import os
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List

import numpy as np
from abacustest.lib_prepare.abacus import ReadInput, WriteInput, AbacusStru
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.constant import RY_TO_EV
from abacusagent.modules.util.comm import run_abacus, generate_work_path, link_abacusjob, collect_metrics
from abacusagent.modules.util.cube_manipulator import read_gaussian_cube, profile1d

def identify_potential_plateaus(
    averaged_potential: List,
    length: float = 10.0,
    threshold: float = 0.01
):
    """
    Identify plateaus in the electrostatic potential using derivative of the electrostatic potential.
    """
    pot_derivatives = []
    n_points = len(averaged_potential)
    stepsize = length / (n_points - 1)
    for i in range(n_points):
        backward_idx = i - 1
        forward_idx = 0 if i == n_points - 1 else i + 1
        pot_derivative = (averaged_potential[forward_idx] - averaged_potential[backward_idx]) / (stepsize * 2.0)
        pot_derivatives.append(pot_derivative)
    
    is_plateau = [abs(deriv) < threshold for deriv in pot_derivatives]

    plateau_ranges = []
    in_plateau, start_idx = False, None
    for i in range(n_points):
        if is_plateau[i] and not in_plateau:
            in_plateau = True
            start_idx = i
        elif not is_plateau[i] and in_plateau:
            in_plateau = False
            if not start_idx == i - 1:
                plateau_ranges.append((start_idx, i-1))
                start_idx = None
        elif is_plateau[i] and i == n_points - 1:
            if not start_idx == i - 1:
                plateau_ranges.append((start_idx, i))
                in_plateau, start_idx = False, None
    
    if len(plateau_ranges) > 0:
        if plateau_ranges[-1][1] == n_points - 1:
            if len(plateau_ranges) > 1:
                combined_plateau = (plateau_ranges[-1][0] - n_points, plateau_ranges[0][1])
                plateau_ranges[0] = combined_plateau
                plateau_ranges.pop()
    
    return plateau_ranges

def calculate_work_functions(averaged_potential: List, fermi_energy, length: float = 10.0):
    """
    Calculate the work function from the averaged electrostatic potential.
    Dipole correction is suppoted and multiple plateau of electrostatic potential can be identified.
    """
    work_function_results = []
    plateau_ranges = identify_potential_plateaus(averaged_potential, length=length, threshold=0.01)
    for plateau_range in plateau_ranges:
        plateau_start, plateau_end = plateau_range
        plateau_averaged_potential = np.average(averaged_potential[plateau_start:plateau_end])
        work_function_results.append({'work_function': plateau_averaged_potential - fermi_energy,
                                      'plateau_start_fractional': plateau_start / len(averaged_potential),
                                      'plateau_end_fractional': plateau_end / len(averaged_potential)})
    
    return work_function_results

def round_coord_pbc1d(a, length):
    a /= length
    return (a - np.floor(a)) * length

def dist_pbc1d(a, b, length):
    a, b = round_coord_pbc1d(a, length), round_coord_pbc1d(b, length)
    if a > b:
        a, b = b, a
    return min(b - a, a + length - b)

def determine_efield_pos_max(stru: AbacusStru, vacuum_direction: Literal['x', 'y', 'z'] = 'z', threshold: float=3.0) -> float:
    """
    Automatically determine the maximum position of the applied saw-shape electric field in dipole correction.
    """
    def calculate_dist(ref_pos, atom_poses, mode=Literal['lower', 'higher']):
        min_dist = None
        if mode == 'lower':
            for atom_pos in atom_poses:
                if atom_pos < ref_pos:
                    dist = dist_pbc1d(ref_pos, atom_pos, length=cell_length)
                    if min_dist is None:
                        min_dist = dist
                    elif dist < min_dist:
                        min_dist = dist
        elif mode == 'higher':
                if atom_pos > ref_pos:
                    dist = dist_pbc1d(ref_pos, atom_pos, length=cell_length)
                    if min_dist is None:
                        min_dist = dist
                    elif dist < min_dist:
                        min_dist = dist
        else:
            raise ValueError("Invalid mode")
        
        return min_dist

    direction_map = {'x': 0, 'y': 1, 'z': 2}
    direction = direction_map[vacuum_direction]
    atom_positions_vac_dir = []
    for atom_idx in range(stru.get_natoms()):
        atom_positions_vac_dir.append(stru.get_coord()[atom_idx][direction])
    
    cell_length = np.linalg.norm(stru.get_cell()[direction]) # Lattice parameter along the given vacuum direction

    if cell_length - max(atom_positions_vac_dir) + min(atom_positions_vac_dir) < threshold:
        # The slab crosses the boundary, and the vacuum lies with in the cell
        stepsize = 1.0 # Angstrom
        # Find lower boundary of the vacuum region
        trial_pos = 1.0 # Angstrom
        lower_boundary = None
        while lower_boundary is None:
            min_dist = calculate_dist(trial_pos, atom_positions_vac_dir, mode='lower')
            if min_dist > threshold:
                lower_boundary = trial_pos
            elif trial_pos + stepsize < cell_length:
                trial_pos += stepsize
            else:
                raise RuntimeError("Unable to find the lower boundary of the vacuum region")
        
        trial_pos = cell_length - 1.0
        upper_boundary = None
        while upper_boundary is None:
            min_dist = calculate_dist(trial_pos, atom_positions_vac_dir, mode='higher')
            if min_dist > threshold:
                upper_boundary = trial_pos
            elif trial_pos - stepsize > 0:
                trial_pos -= stepsize
            else:
                raise RuntimeError("Unable to find the upper boundary of the vacuum region")

        if upper_boundary < lower_boundary - threshold:
            print("There seems have no vacuum region. Check your structure and input arguments")
        elif upper_boundary < lower_boundary:
            print("Warning: The slab is too thick. The vacuum region should be enlarged")
        pos = (upper_boundary + lower_boundary) / 2
    else:
        # The slab does not cross the boundary
        if min(atom_positions_vac_dir) + cell_length - min(atom_positions_vac_dir) < threshold * 2:
            print("Warning: The slab is too close to the boundary. The vacuum region should be enlarged")
        pos = (min(atom_positions_vac_dir) + max(atom_positions_vac_dir) + cell_length) / 2 # Midpoint of leftmost atom (PBC image in the right cell) and rightmost atom
        min_dist = pos - max(atom_positions_vac_dir)
        if pos > cell_length:
            pos -= cell_length
    
    return pos / cell_length, min_dist

def determine_efield_pos_max_scan(stru: AbacusStru, vacuum_direction: Literal['x', 'y', 'z'] = 'z', threshold: float=3.0):
    """
    Using simple scan to automatically determine the maximum position of the applied saw-shape electric field in dipole correction.
    """
    direction_map = {'x': 0, 'y': 1, 'z': 2}
    direction = direction_map[vacuum_direction]
    atom_positions_vac_dir = []
    for atom_idx in range(stru.get_natoms()):
        atom_positions_vac_dir.append(stru.get_coord()[atom_idx][direction])
    
    cell_length = np.linalg.norm(stru.get_cell()[direction]) # Lattice parameter along the given vacuum direction

    scan_pos, scan_stepsize, scan_min_dist_max = 0.0, 0.05, None
    while scan_pos < 1.0:
        min_dist = None
        for atom_idx in range(stru.get_natoms()):
            dist = dist_pbc1d(atom_positions_vac_dir[atom_idx], scan_pos * cell_length, cell_length)
            if min_dist is None:
                min_dist = dist
            elif dist < min_dist:
                min_dist = dist
        
        if scan_min_dist_max is None:
            scan_min_dist_max = min_dist
            scan_min = scan_pos
        elif scan_min_dist_max < min_dist:
            scan_min_dist_max = min_dist
            scan_min = scan_pos
        
        scan_pos += scan_stepsize
    
    if scan_min_dist_max > threshold:
        return scan_min, scan_min_dist_max
    else:
        raise RuntimeError("No suitable position for maximum position of the applied saw-shape electric field found")

def plot_averaged_elecstat_pot(
    averaged_elecstat_data,
    work_path: Path,
    axis: Literal['x', 'y', 'z'] = 'z',
    plot_filename: Optional[str] = "elecstat_pot_profile.png"
) -> Path:
    import matplotlib.pyplot as plt
    plt.plot(averaged_elecstat_data['data'][:, 0], averaged_elecstat_data['data'][:, 1], label='Electrostatic Potential')
    plt.xlim(0, 1)
    plt.xlabel("Fractional Coordinate along " + axis)
    plt.ylabel("Electrostatic Potential (eV)")
    plot_path = os.path.join(work_path, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path

def abacus_cal_work_function(
    abacus_inputs_dir: Path,
    vacuum_direction: Literal['x', 'y', 'z'] = 'z',
    dipole_correction: bool = False,
) -> Dict[str, Any]:
    """
    Calculate the electrostatic potential and work function using ABACUS.
    
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        vacuum_direction (Literal['x', 'y', 'z']): The direction of the vacuum.
        dipole_correction (bool): Whether to apply dipole correction along the vacuum direction.

    Returns:
        A dictionary containing:
        - elecstat_pot_work_function_work_path (Path): Path to the ABACUS job directory calculating electrostatic potential and work function.
        - elecstat_pot_file (Path): Path to the cube file containing the electrostatic potential.
        - averaged_elecstat_pot_plot (Path): Path to the plot of the averaged electrostatic potential.
        - work_function_results (list): A list of dictionary, where each dictionary contains 3 keys:
            - 'work_function': calculated work function
            - 'plateau_start_fractional': Fractional coordinate of start of the identified plateau in the given vacuum direction
            - 'plateau_end_fractional': Fractional coordinate of end of the identified plateau in the given vacuum direction
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacus_inputs_dir,dst=work_path,copy_files=["INPUT", "STRU"], exclude_directories=True)
        input_params = ReadInput(os.path.join(work_path, 'INPUT'))
        stru = AbacusStru.ReadStru(os.path.join(work_path, input_params.get('stru_file', 'STRU')))
        if input_params.get('nspin', 1) not in [1, 2]:
            raise ValueError('Only non spin-polarized and collinear spin-polarized calculation are supported for calculating electrostatic potential and work function')

        input_params['calculation'] = 'scf'
        input_params['out_pot'] = 2

        if dipole_correction:
            input_params['efield_flag'] = 1
            input_params['dip_cor_flag'] = 1

            efield_pos_max, min_dist = determine_efield_pos_max(stru, vacuum_direction)
            input_params['efield_pos_max'] = efield_pos_max
            input_params['efield_pos_dec'] = 0.1
            input_params['efield_amp'] = 0.00
    
        WriteInput(input_params, os.path.join(work_path, 'INPUT'))

        run_abacus(work_path)

        metrics = collect_metrics(work_path, metrics_names=['normal_end', 'converge', 'efermi'])
        if metrics['normal_end'] is not True or metrics['converge'] is not True:
            raise RuntimeError('ABACUS calculation didn\'t end normally or didn\'t reached SCF convergence')

        pot_file = os.path.join(work_path, f"OUT.{input_params.get('suffix', 'ABACUS')}/ElecStaticPot.cube")
        pot = read_gaussian_cube(pot_file)

        profile_result = profile1d(pot, axis=vacuum_direction, average=True)
        profile_result['data'][:, 1] *= RY_TO_EV  # Convert from Rydberg to eV

        direction_map = {'x': 0, 'y': 1, 'z': 2}
        length = np.linalg.norm(stru.get_cell()[direction_map[vacuum_direction]])
        work_function_results = calculate_work_functions(profile_result['data'][:, 1],
                                                         fermi_energy=metrics['efermi'],
                                                         length=length)

        # Plot the averaged electrostatic potential
        plot_path = plot_averaged_elecstat_pot(profile_result, work_path, axis=vacuum_direction)

        return {'elecstat_pot_work_function_work_path': Path(work_path).absolute(),
                'elecstat_pot_file': Path(pot_file).absolute(),
                'averaged_elecstat_pot_plot': Path(plot_path).absolute(),
                'work_function_results': work_function_results}
    except Exception as e:
        return {'message': f"Calculating electrostatic potential and work function failed: {e}"}
