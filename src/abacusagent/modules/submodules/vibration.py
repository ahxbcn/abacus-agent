import os
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
import copy
import json
import math
from itertools import groupby

import numpy as np
from ase.io import read
from ase.vibrations import Vibrations
from ase.calculators.abacus import Abacus, AbacusProfile
from ase.thermochemistry import HarmonicThermo
from ase.io.abacus import read_kpt
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.modules.util.comm import get_physical_cores, generate_work_path, link_abacusjob, run_abacus, collect_metrics

def set_ase_abacus_calculator(abacus_inputs_dir: Path,
                              work_path: Path,
                              extra_input_params: Optional[Dict[str, Any]]) -> Abacus:
    """
    To be deprecated
    Set Abacus calculator using input files in ABACUS input directory. 
    ABACUS will be executed in pure MPI parallalized mode.
    """
    # Parallel settings
    os.environ["OMP_NUM_THREADS"] = "1"
    profile = AbacusProfile(command=f"mpirun -np {get_physical_cores()} abacus")
    out_directory = os.path.join(work_path, "SCF")

    # Read INPUT, STRU
    input_params = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
    input_params.update(extra_input_params)
    stru_file = input_params.get('stru_file', "STRU")
    stru = read(os.path.join(abacus_inputs_dir, stru_file))
    abacus_stru = AbacusStru.ReadStru(os.path.join(abacus_inputs_dir, stru_file))

    # Read KPT
    # TODO: If gamma_only is set and kspacing is not set, absense of KPT file will raise an error
    if 'gamma_only' in input_params.keys():
        kpts = {'gamma_only': input_params['gamma_only']}
    elif 'kspacing' in input_params.keys():
        kpts = {'kspacing': input_params['kspacing']}
    else:
        kpt_file = input_params.get('kpt_file', 'KPT')
        kpt_info = read_kpt(os.path.join(abacus_inputs_dir, kpt_file))
        # Set kpoint information required by `ase.calculators.calculator.kpts2sizeandoffsets`
        # used by ase-abacus
        kpts = {'size': kpt_info['kpts']}
        if kpt_info['mode'] in ['Gamma']:
            kpts['gamma'] = True

    # Get pp and orb from provided STRU file
    pseudo_dir = Path(abacus_inputs_dir).absolute()
    orbital_dir = Path(abacus_inputs_dir).absolute()
    pp_list, orb_list = abacus_stru.get_pp(), abacus_stru.get_orb()
    elements = [key for key, _ in groupby(stru.get_chemical_symbols())]
    pp = {element: ppfile for element, ppfile in zip(elements, pp_list)}
    basis = {element: orbfile for element, orbfile in zip(elements, orb_list)}

    input_params['pseudo_dir'] = pseudo_dir
    input_params['orbital_dir'] = orbital_dir

    calc = Abacus(profile=profile,
                  directory=out_directory,
                  pp=pp,
                  basis=basis,
                  kpts=kpts,
                  **input_params)
    
    return calc

def identify_complex_types(complex_array):
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)

    is_real = np.isclose(imag_part, 0)
    is_pure_imag = np.isclose(real_part, 0) & ~np.isclose(imag_part, 0)
    is_general = ~is_real & ~is_pure_imag

    return is_real, is_pure_imag, is_general

def collect_force(abacusjob_dir):
    metrics = collect_metrics(abacusjob_dir, ['force', 'normal_end', 'converge'])
    if metrics['normal_end'] is not True:
        print(f"ABACUS calculation in {abacusjob_dir} didn't end normally")
    elif metrics['converge'] is not True:
        print(f"ABACUS calculation in {abacusjob_dir} didn't reached SCF convergence")
    else:
        pass

    return metrics['force']

def filter_force(forces, selected_atoms, original_stru):
    """
    Select forces belong to selected atoms
    """
    selected_atoms_force_idx = []
    for selected_atom in selected_atoms:
        selected_atoms_force_idx += [selected_atom*3, selected_atom*3+1, selected_atom*3+2]
    
    filtered_force = np.array(forces)[selected_atoms_force_idx]

    return filtered_force

#@mcp.tool()
def abacus_vibration_analysis(abacus_inputs_dir: Path,
                              selected_atoms: Optional[List[int]] = None,
                              stepsize: float = 0.01,
                              nfree: Literal[2] = 2,
                              temperature: Optional[float] = 298.15):
    """
    Performing vibrational analysis using finite displacement method.
    This tool function is usually followed by a relax calculation (`calculation` is set to `relax`).
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files directory.
        selected_atoms (Optional[List[int]]): Indices of atoms included in the vibrational analysis. If this
            parameter are not given, all atoms in the structure will be included.
        stepsize (float): Step size to displace cartesian coordinates of atoms during the vibrational analysis.
            Units in Angstrom. The default value (0.01 Angstrom) is generally OK.
        nfree (int): Number of force calculations performed for each cartesian coordinate components of each 
            included atom. Now only 2 are supported, where 2 represents calculating matrix element of force constant
            matrix using 3-point center difference and need 2 SCF calculations.
        temperature (float): Temperature used to calculate thermodynamic quantities. Units in Kelvin.
    Returns:
        A dictionary containing the following keys:
        - 'frequencies': List of real frequencies from vibrational analysis. Imaginary frequencies are represented by negative 
            values. Units in cm^{-1}.
        - 'zero_point_energy': Zero-point energy summed over all modes. Units in eV.
        - 'vib_entropy': Vibrational entropy using harmonic approximation. Units in eV/K.
        - 'vib_free_energy': Vibrational Helmholtz free energy using harmonic approximation. Units in eV.
        - 'vib_analysis_work_path': Path to directory performing vibrational analysis. Containing animation of normal modes 
            with non-zero frequency in ASE traj format and `vib` directory containing collected forces.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        if stepsize <= 0:
            raise ValueError("stepsize should be positive.")
        
        work_path = Path(generate_work_path()).absolute()

        input_params = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
        stru_file = input_params.get('stru_file', "STRU")
        stru = AbacusStru.ReadStru(os.path.join(abacus_inputs_dir, stru_file))
        # Provide extra INPUT parameters necessary for vibration analysis using finite difference
        input_params['calculation'] = 'scf'
        input_params['cal_force'] = 1

        vib_cache_dir = os.path.join(work_path, "vib")
        os.makedirs(vib_cache_dir, exist_ok=True)
        vib = Vibrations(stru.to_ase(),
                         name=vib_cache_dir,
                         indices=selected_atoms,
                         delta=stepsize,
                         nfree=nfree)
        
        displaced_stru = copy.deepcopy(stru)
        original_stru_coord = np.array(stru.get_coord(bohr=False, direct=False))

        if selected_atoms is None:
            selected_atoms = [i for i in range(stru.get_natoms())]

        selected_atoms.sort()
        
        DIRECTION_MAP = ['x', 'y', 'z']
        STEP_MAP = {'+': 1, '-': -1}
        disped_stru_job_paths = []
        disped_stru_cache_labels = []

        # Prepare ABACUS input files for the given structure
        abacus_scf_work_path = os.path.join(work_path, "SCF")
        original_stru_job_path = os.path.join(abacus_scf_work_path, "eq")
        os.makedirs(original_stru_job_path)

        link_abacusjob(src=abacus_inputs_dir,
                       dst=original_stru_job_path,  
                       copy_files=['INPUT', 'STRU', 'KPT', 'abacus.log'],
                       exclude_directories=True)
        WriteInput(input_params, os.path.join(original_stru_job_path, "INPUT"))
        disped_stru_job_paths.append(original_stru_job_path)
        disped_stru_cache_labels.append(os.path.join(vib_cache_dir, 'cache.eq.json'))

        # Prepare ABACUS input files for each displaced structure. nfree is assumed to be 2.
        for selected_atom in selected_atoms:
            for direction in range(3): # x, y and z directions
                displaced_stru_coord = copy.deepcopy(original_stru_coord)
                for step in STEP_MAP.keys(): # Two steps along one direction
                    disped_stru_job_path = os.path.join(abacus_scf_work_path, f"disp-{selected_atom}-{DIRECTION_MAP[direction]}{step}")
                    os.makedirs(disped_stru_job_path)

                    link_abacusjob(src=abacus_inputs_dir,
                                   dst=disped_stru_job_path, 
                                   copy_files=['INPUT', 'STRU', 'KPT', 'abacus.log'],
                                   exclude_directories=True)
                    
                    displaced_stru_coord[selected_atom][direction] = original_stru_coord[selected_atom][direction] + stepsize * STEP_MAP[step]
                    displaced_stru.set_coord(displaced_stru_coord, bohr=False, direct=False)
                    WriteInput(input_params, os.path.join(disped_stru_job_path, "INPUT"))
                    displaced_stru.write(os.path.join(disped_stru_job_path, stru_file))

                    disped_stru_job_paths.append(disped_stru_job_path)
                    disped_stru_cache_labels.append(os.path.join(vib_cache_dir, f"cache.{selected_atom}{DIRECTION_MAP[direction]}{step}.json"))
        
        # Run ABACUS calculation for all prepared jobs
        run_abacus(disped_stru_job_paths)
        
        # Collect needed forces for each ABACUS calculation and dump to cache directory used by ase.vibrations
        cache_forces_json = {"forces": {
            "__ndarray__": [[len(selected_atoms), 3],
                            "float64",
                            []]
        }}
        for disped_stru_job_path, disped_stru_cache_label in list(zip(disped_stru_job_paths, disped_stru_cache_labels)):
            force = collect_force(disped_stru_job_path)
            cache_forces_json["forces"]["__ndarray__"][2] = list(filter_force(force, selected_atoms, stru))
            with open(os.path.join(vib_cache_dir, disped_stru_cache_label), "w") as fin:
                json.dump(cache_forces_json, fin)
        
        # Do the vibration analysis
        vib.summary()
        # Generate list of frequencies in the return value
        frequencies = vib.get_frequencies()
        real_freq_mask, imag_freq_mask, complex_freq_mask = identify_complex_types(frequencies)
        real_freq, imag_freq = np.real(frequencies[real_freq_mask]).tolist(), frequencies[imag_freq_mask].tolist()
        for key, value in enumerate(imag_freq):
            imag_freq[key] = -math.fabs(value.imag) # Represent imaginary frequency with negative number
        freqs = imag_freq + real_freq

        # Write animations of normal modes in ASE traj format
        vib.write_mode()

        # Thermochemistry calculations
        # Vibrations.get_energies() gets `h \nu` for each mode, which is from the eigenvalues of force constant
        # matrix. The force constant matrix should be a real symmetric matrix mathematically, but due to numerical
        # errors during calculating its matrix element, it will deviate from symmetric matric slightly, and its eigenvalue
        # will have quite small imaginary parts. Magnitude of imaginary parts will decrease as the calculation accuracy
        # increases, and it's safe to use norm of the complex eigenvalue as vibration energy if the calculation is 
        # accurate enough.
        vib_energies = vib.get_energies()
        vib_energies_float = [float(np.linalg.norm(i)) for i in vib_energies]
        zero_point_energy = sum(vib_energies_float) / 2
        thermo = HarmonicThermo(vib_energies, ignore_imag_modes=True)
        entropy = thermo.get_entropy(temperature)
        free_energy = thermo.get_helmholtz_energy(temperature)

        return {'frequencies': freqs,
                'zero_point_energy': float(zero_point_energy),
                'vib_entropy': float(entropy),
                'vib_free_energy': float(free_energy),
                'vib_analysis_work_dir': Path(work_path).absolute()}
    except Exception as e:
        print(e)
        return {'message': f"Doing vibration analysis failed: {e}"}

