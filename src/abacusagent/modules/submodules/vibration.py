import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from itertools import groupby

from ase.io import read
from ase.calculators.abacus import Abacus, AbacusProfile
from ase.io.abacus import read_kpt
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput
from abacustest.lib_model.comm import check_abacus_inputs
from abacustest.lib_model.model_019_vibration import prepare_abacus_vibration_analysis, post_abacus_vibration_analysis_onejob

from abacusagent.modules.util.comm import get_physical_cores, generate_work_path, link_abacusjob, run_abacus

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

def abacus_vibration_analysis(abacus_inputs_dir: Path,
                              selected_atoms: Optional[List[int]] = None,
                              stepsize: float = 0.01,
                              temperature: Optional[float] = 298.15):
    """
    Performing vibrational analysis using finite displacement method.
    This tool function is usually followed by a relax calculation (`calculation` is set to `relax`).
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files directory.
        selected_atoms (Optional[List[int]]): Indices (started from 1) of atoms included in the vibrational analysis. If this
            parameter are not given, all atoms in the structure will be included.
        stepsize (float): Step size to displace cartesian coordinates of atoms during the vibrational analysis.
            Units in Angstrom. The default value (0.01 Angstrom) is generally OK.
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
        link_abacusjob(src=abacus_inputs_dir,
                       dst=work_path,
                       copy_files=["INPUT", "STRU", "KPT"],
                       exclude=["OUT.*", "*.log", "*.out", "*.json", "log"],
                       exclude_directories=True)
        job_paths = prepare_abacus_vibration_analysis(job_path=work_path,
                                                      selected_atoms=selected_atoms,
                                                      stepsize=stepsize)
        
        # Run ABACUS calculation for all prepared jobs
        run_abacus(job_paths)
        
        vib_results = post_abacus_vibration_analysis_onejob(work_path, temperature=temperature)
        vib_results['vib_analysis_work_dir'] = Path(work_path).absolute()
        return vib_results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'message': f"Doing vibration analysis failed: {e}"}

