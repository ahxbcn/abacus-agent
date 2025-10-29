from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.band import abacus_cal_band as _abacus_cal_band

@mcp.tool()
def abacus_cal_band(abacus_inputs_dir: Path,
                    mode: Literal["nscf", "pyatb", "auto"] = "auto",
                    kpath: Union[List[str], List[List[str]]] = None,
                    high_symm_points: Dict[str, List[float]] = None,
                    energy_min: float = -10,
                    energy_max: float = 10,
                    insert_point_nums: int = 30
) -> Dict[str, float|str]:
    """
    Calculate band using ABACUS based on prepared directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    PYATB or ABACUS NSCF calculation will be used according to parameters in INPUT.
    Args:
        abacus_inputs_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
        mode: Method used to plot band. Should be `auto`, `pyatb` or `nscf`. 
            - `nscf` means using `nscf` calculation in ABACUS to calculate and plot the band
            - `pyatb` means using PYATB to plot the band
            - `auto` means deciding use `nscf` or `pyatb` mode according to the `basis_type` in INPUT file and files included in `abacus_inputs_dir`.
                -- If charge files are in `abacus_input_dir`, `nscf` mode will be used.
                -- If matrix files are in `abacus_input_dir`, `pyatb` mode will be used.
                -- If no matrix file or charge file are in `abacus_input_dir`, will determine mode by `basis_type`. If `basis_type` is lcao, will use `pyatb` mode.
                    If `basis_type` is pw, will use `nscf` mode.
        kpath (Tuple[List[str], List[List[str]]]): 
                A list of name of high symmetry points in the band path. Non-continuous line of high symmetry points are stored as seperate lists.
                For example, ['G', 'M', 'K', 'G'] and [['G', 'X', 'P', 'N', 'M', 'S'], ['S_0', 'G', R']] are both acceptable inputs.
                Default is None. If None, will use automatically generated kpath.
                `kpath` must be used with `high_symm_points` to take effect.
        high_symm_points: A dictionary containing high symmetry points and their coordinates in the band path. All points in `kpath` should be included.
                For example, {'G': [0, 0, 0], 'M': [0.5, 0.0, 0.0], 'K': [0.33333333, 0.33333333, 0.0], 'G': [0, 0, 0]}.
                Default is None. If None, will use automatically generated high symmetry points.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
        insert_point_nums (int): Number of points to insert between two high symmetry points. Default is 30.
    Returns:
        A dictionary containing band gap, path to the work directory for calculating band and path to the plotted band.
    Raises:
    """
    return _abacus_cal_band(abacus_inputs_dir, mode, kpath, high_symm_points, energy_min, energy_max, insert_point_nums)

@mcp.tool()
def get_high_symm_points(abacusjob_dir: Path) -> Dict[str, Any]:
    """
    Get high symmetry points and kpath for STRU file in ABACUS inputs directory.
    Args:
        abacusjob_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    Returns:
        A dictionary containing high symmetry points and suggested kpath for STRU file in ABACUS inputs directory. The most important keys are:
        - path (List[List[str]]): Suggested path for the given structure.
        - point_coords: Coordinates of high symmetry points in reciprocal space.
    """
    from abacusagent.modules.submodules.band import get_high_symm_points_from_abacus_inputs_dir

    return get_high_symm_points_from_abacus_inputs_dir(abacusjob_dir)

#@mcp.tool()
def cal_band(stru_file: Path,
             stru_type: Literal['cif', 'poscar', 'abacus/stru'],
             lcao: bool = True,
             nspin: Literal[1, 2] = 1,
             #soc: bool=False,
             dftu: bool = False,
             dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
             init_mag: Optional[Dict[str, float]] = None,
             relax: bool = False,
             max_relax_attempts: int = 3,
             relax_cell: bool = False,
             force_thr_ev: Optional[float] = None,
             stress_thr_kbar: Optional[float] = None,
             max_steps: Optional[int] = None,
             relax_method: Optional[Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"]] = None,
             fixed_axes: Optional[Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"]] = None,
             relax_new: Optional[bool] = None,
             pressure: List[float] = [0.0, 0.0, 0.0],
             mode: Literal["nscf", "pyatb", "auto"] = "auto",
             energy_min: float = -10,
             energy_max: float = 10
) -> Dict[str, Any]:
    """
    Calculate the band of the given structure.

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
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict): The DFT+U parameters, should be 'auto' or a dict
            If dft_param is set to 'auto', hubbard U parameters will be set to d-block and f-block elements automatically. For d-block elements, default U=4eV will
                be set to d orbital. For f-block elements, default U=6eV will be set to f orbital.
            If dft_param is a dict, the keys should be name of elements and the value has two choices:
                - A float number, which is the Hubbard U value of the element. The corrected orbital will be infered from the name of the element.
                - A list containing two elements: the corrected orbital (should be 'p', 'd' or 'f') and the Hubbard U value.
                For example, {"Fe": ["d", 4], "O": ["p", 1]} means applying DFT+U to Fe 3d orbital with U=4 eV and O 2p orbital with U=1 eV.
        init_mag ( dict or None): The initial magnetic moment for magnetic elements, should be a dict like {"Fe": 4, "Ti": 1}, where the key is the element symbol and the value is the initial magnetic moment.
        relax (bool): Whether to relax the structure.
        force_thr_ev: Force convergence threshold in eV/Å, default is 0.01.
        stress_thr_kbar: Stress convergence threshold in kbar, default is 1.0, this is only used when relax_cell is True.
        max_steps: Maximum number of relaxation steps, default is 100.
        relax_cell: Whether to relax the cell parameters, default is False.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes  
        relax_method: The relaxation method to use, can be 'cg', 'bfgs', 'bfgs_trad', 'cg_bfgs', 'sd', or 'fire'. Default is 'cg'.
        relax_new: If use new implemented CG method, default is True.
        pressure (List[float]): Pressure to apply. Units in kbar.
        mode: Method used to plot band. Should be `auto`, `pyatb` or `nscf`. 
            - `nscf` means using `nscf` calculation in ABACUS to calculate and plot the band
            - `pyatb` means using PYATB to plot the band
            - `auto` means deciding use `nscf` or `pyatb` mode according to the `basis_type` in INPUT file and files included in `abacus_inputs_dir`.
                -- If charge files are in `abacus_input_dir`, `nscf` mode will be used.
                -- If matrix files are in `abacus_input_dir`, `pyatb` mode will be used.
                -- If no matrix file or charge file are in `abacus_input_dir`, will determine mode by `basis_type`. If `basis_type` is lcao, will use `pyatb` mode.
                    If `basis_type` is pw, will use `nscf` mode.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
    """
    from abacusagent.modules.submodules.abacus import abacus_prepare
    from abacusagent.modules.submodules.relax import abacus_do_relax

    extra_inputs = {'press1': pressure[0], 'press2': pressure[1], 'press3': pressure[2]}
    prepare_outputs = abacus_prepare(
        stru_file,
        stru_type,
        'scf', #job_type
        lcao,
        nspin,
        False, #soc
        dftu,
        dftu_param,
        init_mag,
        False, #afm
        extra_inputs,
    )

    if 'message' not in prepare_outputs.keys():
        abacus_inputs_dir = prepare_outputs['abacus_inputs_dir']
    else:
        raise Exception(f"Prepare ABACUS inputs failed: {prepare_outputs['message']}")
    
    if relax:
        relaxed = False
        relax_attempt = 0
        while not relaxed and relax_attempt < max_relax_attempts:
            relax_attempt += 1
            relax_outputs = abacus_do_relax(
                abacus_inputs_dir,
                force_thr_ev,
                stress_thr_kbar,
                max_steps,
                relax_cell,
                fixed_axes,
                relax_method,
                relax_new,
            )
            if relax_outputs['result']['relax_converge'] is True:
                relaxed = True
                abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
            elif 'message' not in relax_outputs.keys():
                abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
            else:
                raise Exception(f"Relax failed: {relax_outputs['message']}")

    band_calculation_outputs = _abacus_cal_band(
        abacus_inputs_dir,
        mode=mode,
        energy_min=energy_min,
        energy_max=energy_max,
    )
    
    return band_calculation_outputs
