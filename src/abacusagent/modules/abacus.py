import os
import json
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union

import numpy as np

from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import generate_work_path, run_abacus, collect_metrics
from abacusagent.modules.submodules.abacus import abacus_prepare as _abacus_prepare
from abacusagent.modules.submodules.abacus import abacus_collect_data as _abacus_collect_data
from abacusagent.modules.submodules.abacus import abacus_modify_input as _abacus_modify_input
from abacusagent.modules.submodules.abacus import abacus_modify_stru as _abacus_modify_stru


@mcp.tool()
def abacus_prepare(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    #pp_path: Optional[str] = None,
    #orb_path: Optional[str] = None,
    job_type: Literal["scf", "relax", "cell-relax", "md"] = "scf",
    lcao: bool = True,
    nspin: Literal[1, 2, 4] = 1,
    soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    afm: bool = False,
    extra_input: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Prepare mandatory input files for ABACUS calculation from a structure file.
    This function does not perform any actual calculation, but is necessary to use this function
    to prepare a directory containing necessary input files for ABACUS calculation. 
    If user provides ABACUS input files by him/herself, this function should not be used.
    
    Args:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        job_type (Literal["scf", "relax", "cell-relax", "md"] = "scf"): The type of job to be performed, can be:
            'scf': Self-consistent field calculation, which is the default. 
            'relax': Geometry relaxation calculation, which will relax the atomic position to the minimum energy configuration.
            'cell-relax': Cell relaxation calculation, which will relax the cell parameters and atomic positions to the minimum energy configuration.
            'md': Molecular dynamics calculation, which will perform molecular dynamics simulation.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized), or 4 (non-collinear spin). Default is 1.
        soc (bool): Whether to use spin-orbit coupling, if True, nspin should be 4.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict): The DFT+U parameters, should be 'auto' or a dict
            If dft_param is set to 'auto', hubbard U parameters will be set to d-block and f-block elements automatically. For d-block elements, default U=4eV will
                be set to d orbital. For f-block elements, default U=6eV will be set to f orbital.
            If dft_param is a dict, the keys should be name of elements and the value has two choices:
                - A float number, which is the Hubbard U value of the element. The corrected orbital will be infered from the name of the element.
                - A list containing two elements: the corrected orbital (should be 'p', 'd' or 'f') and the Hubbard U value.
                For example, {"Fe": ["d", 4], "O": ["p", 1]} means applying DFT+U to Fe 3d orbital with U=4 eV and O 2p orbital with U=1 eV.
        init_mag ( dict or None): The initial magnetic moment for magnetic elements, should be a dict like {"Fe": 4, "Ti": 1}, where the key is the element symbol and the value is the initial magnetic moment.
        afm (bool): Whether to use antiferromagnetic calculation, default is False. If True, half of the magnetic elements will be set to negative initial magnetic moment.
        extra_input: Extra input parameters in the prepared INPUT file. 
    
    Returns:
        A dictionary containing the job path.
        - 'abacus_inputs_dir': The absolute path to the generated ABACUS input directory, containing INPUT, STRU, pseudopotential and orbital files.
        - 'input_content': The content of the generated INPUT file.
    Raises:
        FileNotFoundError: If the structure file or pseudopotential path does not exist.
        ValueError: If LCAO basis set is selected but no orbital library path is provided.
        RuntimeError: If there is an error preparing input files.
    """
    return _abacus_prepare(stru_file, stru_type, job_type, lcao, nspin, soc, dftu, dftu_param, init_mag, afm, extra_input)

@mcp.tool()
def abacus_modify_input(
    abacus_inputs_dir: Path,
    dft_plus_u_settings: Optional[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]]] = None,
    extra_input: Optional[Dict[str, Any]] = None,
    remove_input: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Modify keywords in ABACUS INPUT file.
    Args:
        abacus_inputs_dir (str): Path to the directory containing the ABACUS input files.
        dft_plus_u_setting: Dictionary specifying DFT+U settings.  
            - Key: Element symbol (e.g., 'Fe', 'Ni').  
            - Value: A list with one or two elements:  
                - One-element form: float, representing the Hubbard U value (orbital will be inferred).  
                - Two-element form: [orbital, U], where `orbital` is one of {'p', 'd', 'f'}, and `U` is a float.
        extra_input: Additional key-value pairs to update the INPUT file. If the name of the key is already in the INPUT file, the value will be updated.
        remove_input: A list of parameter names to be removed in the INPUT file

    Returns:
        A dictionary containing:
        - modified_abacus_inputs_dir: the path of the modified INPUT file.
        - input_content: the content of the modified INPUT file as a dictionary.
    Raises:
        FileNotFoundError: If path of given INPUT file does not exist
        RuntimeError: If write modified INPUT file failed
    """
    
    return _abacus_modify_input(abacus_inputs_dir, dft_plus_u_settings, extra_input, remove_input)

@mcp.tool()
def abacus_modify_stru(
    abacus_inputs_dir: Path,
    pp: Optional[Dict[str, str]] = None,
    orb: Optional[Dict[str, str]] = None,
    fix_atoms_idx: Optional[List[int]] = None,
    cell: Optional[List[List[float]]] = None,
    coord_change_type: Literal['scale', 'original'] = 'scale',
    movable_coords: Optional[List[bool]] = None,
    initial_magmoms: Optional[List[float]] = None,
    angle1: Optional[List[float]] = None,
    angle2: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Modify pseudopotential, orbital, atom fixation, initial magnetic moments and initial velocities in ABACUS STRU file.
    Args:
        abacus_inputs_dir (str): Path to the directory containing the ABACUS input files.
        pp: Dictionary mapping element names to pseudopotential file paths.
            If not provided, the pseudopotentials from the original STRU file are retained.
        orb: Dictionary mapping element names to numerical orbital file paths.
            If not provided, the orbitals from the original STRU file are retained.
        fix_atoms_idx: List of indices of atoms to be fixed.
        cell: New cell parameters to be set in the STRU file. Should be a list of 3 lists, each containing 3 floats.
        coord_change_type: Type of coordinate change to apply.
            - 'scale': Scale the coordinates by the cell parameters. Suitable for most cases.
            - 'original': Use the original coordinates without scaling. Suitable for single atom or molecule in a large cell.
        movable_coords: For each fixed atom, specify which coordinates are allowed to move.
            Each entry is a list of 3 integers (0 or 1), where 1 means the corresponding coordinate (x/y/z) can move.
            Example: if `fix_atoms_idx = [1]` and `movable_coords = [[0, 1, 1]]`, the x-coordinate of atom 1 will be fixed.
        initial_magmoms: Initial magnetic moments for atoms.
            - For collinear calculations: a list of floats, shape (natom).
            - For non-collinear using Cartesian components: a list of 3-element lists, shape (natom, 3).
            - For non-collinear using angles: a list of floats, shape (natom), one magnetude of magnetic moment per atom.
        angle1: in non-colinear case, specify the angle between z-axis and real spin, in angle measure instead of radian measure
        angle2: in non-colinear case, specify angle between x-axis and real spin in projection in xy-plane , in angle measure instead of radian measure

    Returns:
        A dictionary containing:
        - modified_abacus_inputs_dir: the path of the modified ABACUS STRU file
        - stru_content: the content of the modified ABACUS STRU file as a string.
    Raises:
        ValueError: If `stru_file` is not path of a file, or dimension of initial_magmoms, angle1 or angle2 is not equal with number of atoms,
          or length of fixed_atoms_idx and movable_coords are not equal, or element in movable_coords are not a list with 3 bool elements
        KeyError: If pseudopotential or orbital are not provided for a element
    """
    return _abacus_modify_stru(abacus_inputs_dir, pp, orb, fix_atoms_idx, cell, coord_change_type, movable_coords, initial_magmoms, angle1, angle2)

@mcp.tool()
def abacus_collect_data(
    abacus_outputs_dir: Path,
    metrics: List[Literal["version", "ncore", "omp_num", "normal_end", "INPUT", "kpt", "fft_grid",
                          "nbase", "nbands", "nkstot", "ibzk", "natom", "nelec", "nelec_dict", "point_group",
                          "point_group_in_space_group", "converge", "total_mag", "absolute_mag", "energy", 
                          "energy_ks", "energies", "volume", "efermi", "energy_per_atom", "force", "forces", 
                          "stress", "virial", "pressure", "stresses", "virials", "pressures", "largest_gradient", "largest_gradient_stress",
                          "band", "band_weight", "band_plot", "band_gap", "total_time", "stress_time", "force_time", 
                          "scf_time", "scf_time_each_step", "step1_time", "scf_steps", "atom_mags", "atom_mag", 
                          "atom_elec", "atom_orb_elec", "atom_mag_u", "atom_elec_u", "drho", "drho_last", 
                          "denergy", "denergy_last", "denergy_womix", "denergy_womix_last", "lattice_constant", 
                          "lattice_constants", "cell", "cells", "cell_init", "coordinate", "coordinate_init", 
                          "element", "label", "element_list", "atomlabel_list", "pdos", "charge", "charge_spd", 
                          "atom_mag_spd", "relax_converge", "relax_steps", "ds_lambda_step", "ds_lambda_rms", 
                          "ds_mag", "ds_mag_force", "ds_time", "mem_vkb", "mem_psipw"]]
                          = ["normal_end", "converge", "energy", "total_time"]
) -> Dict[str, Any]:
    """
    Collect results of the given metrics listed below after ABACUS calculation.
    name of collected metrics must be selected from the list below.

    Args:
        abacus_outputs_dir (str): Path to the directory containing the ABACUS job output files. 
        metrics (List[str]): List of metric names to collect.  
                  metric_name  description
                      version: the version of ABACUS
                        ncore: the mpi cores
                      omp_num: the omp cores
                   normal_end: if the job is normal ending
                        INPUT: a dict to store the setting in OUT.xxx/INPUT, see manual of ABACUS INPUT file
                          kpt: list, the K POINTS setting in KPT file
                     fft_grid: fft grid for charge/potential
                        nbase: number of basis in LCAO
                       nbands: number of bands
                       nkstot: total K point number
                         ibzk: irreducible K point number
                        natom: total atom number
                        nelec: total electron number
                   nelec_dict: dict of electron number of each species
                  point_group: point group
   point_group_in_space_group: point group in space group
                     converge: if the SCF is converged
                    total_mag: total magnetism (Bohr mag/cell)
                 absolute_mag: absolute magnetism (Bohr mag/cell)
                       energy: the total energy (eV)
                    energy_ks: the E_KohnSham, unit in eV
                     energies: list of total energy of each ION step
                       volume: the volume of cell, in A^3
                       efermi: the fermi energy (eV). If has set nupdown, this will be a list of two values. The first is up, the second is down.
              energy_per_atom: the total energy divided by natom, (eV)
                        force: list[3*natoms], force of the system, if is MD or RELAX calculation, this is the last one
                       forces: list of force, the force of each ION step. Dimension is [nstep,3*natom]
                       stress: list[9], stress of the system, if is MD or RELAX calculation, this is the last one
                       virial: list[9], virial of the system, = stress * volume, and is the last one.
                     pressure: the pressure of the system, unit in kbar.
                     stresses: list of stress, the stress of each ION step. Dimension is [nstep,9]
                      virials: list of virial, the virial of each ION step. Dimension is [nstep,9]
                    pressures: list of pressure, the pressure of each ION step.
             largest_gradient: list, the largest gradient of each ION step. Unit in eV/Angstrom
      largest_gradient_stress: list, the largest stress of each ION step. Unit in kbar
                         band: Band of system. Dimension is [nspin,nk,nband].
                  band_weight: Band weight of system. Dimension is [nspin,nk,nband].
                    band_plot: Will plot the band structure. Return the file name of the plot.
                     band_gap: band gap of the system
                   total_time: the total time of the job
                  stress_time: the time to do the calculation of stress
                   force_time: the time to do the calculation of force
                     scf_time: the time to do SCF
           scf_time_each_step: list, the time of each step of SCF
                   step1_time: the time of 1st SCF step
                    scf_steps: the steps of SCF
                    atom_mags: list of list, the magnization of each atom of each ion step.
                     atom_mag: list, the magnization of each atom. Only the last ION step.
                    atom_elec: list of list of each atom. Each atom list is a list of each orbital, and each orbital is a list of each spin
                atom_orb_elec: list of list of each atom. Each atom list is a list of each orbital, and each orbital is a list of each spin
                   atom_mag_u: list of a dict, the magnization of each atom calculated by occupation number. Only the last SCF step.
                  atom_elec_u: list of a dict with keys are atom index, atom label, and electron of U orbital.
                         drho: [], drho of each scf step
                    drho_last: drho of the last scf step
                      denergy: [], denergy of each scf step
                 denergy_last: denergy of the last scf step
                denergy_womix: [], denergy (calculated by rho without mixed) of each scf step
           denergy_womix_last: float, denergy (calculated by rho without mixed) of last scf step
             lattice_constant: a list of six float which is a/b/c,alpha,beta,gamma of cell. If has more than one ION step, will output the last one.
            lattice_constants: a list of list of six float which is a/b/c,alpha,beta,gamma of cell
                         cell: [[],[],[]], two-dimension list, unit in Angstrom. If is relax or md, will output the last one.
                        cells: a list of [[],[],[]], which is a two-dimension list of cell vector, unit in Angstrom.
                    cell_init: [[],[],[]], two-dimension list, unit in Angstrom. The initial cell
                   coordinate: [[],..], two dimension list, is a cartesian type, unit in Angstrom. If is relax or md, will output the last one
              coordinate_init: [[],..], two dimension list, is a cartesian type, unit in Angstrom. The initial coordinate
                      element: list[], a list of the element name of all atoms
                        label: list[], a list of atom label of all atoms
                 element_list: same as element
               atomlabel_list: same as label
                         pdos: a dict, keys are 'energy' and 'orbitals', and 'orbitals' is a list of dict which is (index,species,l,m,z,data), dimension of data is nspin*ne
                       charge: list, the charge of each atom.
                   charge_spd: list of list, the charge of each atom spd orbital.
                 atom_mag_spd: list of list, the magnization of each atom spd orbital.
               relax_converge: if the relax is converged
                  relax_steps: the total ION steps
               ds_lambda_step: a list of DeltaSpin converge step in each SCF step
                ds_lambda_rms: a list of DeltaSpin RMS in each SCF step
                       ds_mag: a list of list, each element list is for each atom. Unit in uB
                 ds_mag_force: a list of list, each element list is for each atom. Unit in eV/uB
                      ds_time: a list of the total time of inner loop in deltaspin for each scf step.
                      mem_vkb: the memory of VNL::vkb, unit it MB
                    mem_psipw: the memory of PsiPW, unit it MB

    Returns:
        A dictionary containing all collected metrics
    Raises:
        IOError: If read abacus result failed
        RuntimeError: If error occured during collectring data using abacustest
    """
    return _abacus_collect_data(abacus_outputs_dir, metrics)
