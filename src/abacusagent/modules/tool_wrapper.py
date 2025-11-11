from pathlib import Path
import os
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import get_relax_precision
from abacusagent.modules.submodules.abacus import abacus_prepare
from abacusagent.modules.submodules.scf import abacus_calculation_scf as _abacus_calculation_scf
from abacusagent.modules.submodules.cube import abacus_cal_elf
from abacusagent.modules.submodules.band import abacus_cal_band as _abacus_cal_band
from abacusagent.modules.submodules.bader import abacus_badercharge_run as _abacus_badercharge_run
from abacusagent.modules.submodules.dos import abacus_dos_run as _abacus_dos_run
from abacusagent.modules.submodules.phonon import abacus_phonon_dispersion as _abacus_phonon_dispersion
from abacusagent.modules.submodules.elastic import abacus_cal_elastic as _abacus_cal_elastic
from abacusagent.modules.submodules.eos import abacus_eos
from abacusagent.modules.submodules.relax import abacus_do_relax as _abacus_do_relax
from abacusagent.modules.submodules.md import abacus_run_md
from abacusagent.modules.submodules.work_function import abacus_cal_work_function
from abacusagent.modules.submodules.vacancy import abacus_cal_vacancy_formation_energy


def prepare_abacus_inputs(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2, 4] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
) -> Dict[str, Any]:
    """
    Commom prepare ABACUS inputs for ABACUS calculation in this file.
    """
    extra_input = {}
    if dft_functional in ['PBE', 'PBEsol', 'LDA', 'SCAN']:
        extra_input['dft_functional'] = dft_functional
    elif dft_functional in ['R2SCAN']:
        extra_input['dft_functional'] = 'MGGA_X_R2SCAN+MGGA_C_R2SCAN'
    elif dft_functional in [ 'HSE', 'PBE0']:
        print("Calculating with hybird functionals like HSE and PBE0 needs much longer time and much more meory than GGA functionals such as PBE.")
        extra_input['dft_functional'] = dft_functional
        os.environ['ABACUS_COMMAND'] = "OMP_NUM_THREADS=16 abacus" # Set to use OpenMP for hybrid functionals like HSE and PBE0
    else:
        print("DFT functional not supported now. Use dafault PBE functional.")
    
    abacus_prepare_outputs = abacus_prepare(stru_file=stru_file,
                                            stru_type=stru_type,
                                            lcao=lcao,
                                            nspin=nspin,
                                            #soc=soc,
                                            dftu=dftu,
                                            dftu_param=dftu_param,
                                            init_mag=init_mag,
                                            #afm=afm,
                                            extra_input=extra_input)
    
    abacus_inputs_dir = abacus_prepare_outputs['abacus_inputs_dir']
    
    return abacus_inputs_dir

def do_relax(
    abacus_inputs_dir: Path = None,
    max_steps: int = 100,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    fixed_axes: Optional[Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"]] = None,
    relax_method: Optional[Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"]] = None,
) -> Dict[str, Any]:
    """
    Do relax calculation using ABACUS.
    """
    relax_thresholds = get_relax_precision(relax_precision)
    
    if relax_cell is False: # For ABACUS LTSv3.10.0
        relax_method = 'bfgs_trad'
    else:
        relax_method = 'cg'
    
    relax_outputs = _abacus_do_relax(abacus_inputs_dir,
                                     force_thr_ev=relax_thresholds['force_thr_ev'],
                                     stress_thr_kbar=relax_thresholds['stress_thr'],
                                     max_steps=max_steps,
                                     relax_cell=relax_cell,
                                     relax_method=relax_method,
                                     fixed_axes=fixed_axes)
    
    if relax_outputs['result']['normal_end'] is False:
        raise ValueError('Relaxation calculation failed')
    elif relax_outputs['result']['relax_converge'] is False:
        return {"msg":f'Relaxation calculation did not converge in {max_steps} steps',
                "final_stru": Path(relax_outputs['new_abacus_inputs_dir']) / "STRU",
                **relax_outputs["result"]}
    else:
        print("Relax calculation completed successfully.")
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    return {'final_stru': Path(relax_outputs['new_abacus_inputs_dir']) / "STRU",
            **relax_outputs}

@mcp.tool()
def run_abacus_calculation(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    property: Literal['bader_charge', 'elf', 'band', 'dos', 'elastic_properties', 'eos', 'phonon_dispersion', 'md',
                      'work_function', 'vacancy_formation_energy'] = 'bader_charge',
    lcao: bool = True,
    nspin: Literal[1, 2, 4] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
    vacuum_direction: Optional[Literal['x', 'y', 'z']] = 'z',
    dipole_correction: bool = False,
    vacancy_supercell: List[int] = [1, 1, 1],
    vacancy_element: str = None,
    vacancy_element_index: int = 1,
    vacancy_relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    md_type: Literal['nve', 'nvt', 'npt', 'langevin'] = 'nve',
    md_nstep: int = 10,
    md_dt: float = 1.0,
    md_tfirst: float = 300.0,
    md_tlast: float = 300.0,
    md_thermostat: Literal['nhc', 'anderson', 'berendsen', 'rescaling', 'rescale_v'] = 'nhc',
    md_pmode: Literal['iso', 'aniso', 'tri'] = 'iso',
    md_pcouple: Literal['none', 'xy', 'xz', 'yz', 'xyz'] = 'none',
    md_dumpfreq: int = 1,
    md_seed: int = -1
):
    """
    Calculate properties using ABACUS.

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        property: String indicating the property to calculate, can be 'bader_charge', 'elf', 'band', 'dos', 'elastic_properties', 'eos', 'phonon_dispersion', or 'md'. Default is 'bader_charge'.
            For band and dos calculations, only nspin=1 or 2 is supported.
            For equation of state fitting, only cubic cell is supported.
            For phonon dispersion, the thermodynamic properties at 298.15 K is calculated.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized), or 4 (non-collinear spin). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict): The DFT+U parameters, should be 'auto' or a dict
            If dft_param is set to 'auto', hubbard U parameters will be set to d-block and f-block elements automatically. For d-block elements, default U=4eV will
                be set to d orbital. For f-block elements, default U=6eV will be set to f orbital.
            If dft_param is a dict, the keys should be name of elements and the value has two choices:
                - A float number, which is the Hubbard U value of the element. The corrected orbital will be infered from the name of the element.
                - A list containing two elements: the corrected orbital (should be 'p', 'd' or 'f') and the Hubbard U value.
                For example, {"Fe": ["d", 4], "O": ["p", 1]} means applying DFT+U to Fe 3d orbital with U=4 eV and O 2p orbital with U=1 eV.
        init_mag ( dict or None): The initial magnetic moment for magnetic elements, should be a dict like {"Fe": 4, "Ti": 1}, where the key is the element symbol and the value is the initial magnetic moment.

        The following parameters are only used when `property` is `work_function`:
        vacuum_direction (Literal['x', 'y', 'z'] or None): The direction of the vacuum layer. Can be 'x', 'y', or 'z'. Default is 'z'.
        dipole_correction (bool): Whether to apply dipole correction during the calculation of work function. Default is False.
        
        The following parameters are only used when `property` is `vacancy_formation_energy`:
        vacancy_supercell (List[int]): Supercell matrix. Defaults to [1, 1, 1], which means no supercell in the calculation of vacancy formation energy.
        vacancy_element (str): Element to be removed. Default is None, which means the first type of element in the structure file.
        vacancy_element_index (int): Index of the vacancy element. Defaults to 1. The index is in the original structure. and should be counted for the given element.
        vacancy_relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation for the calculation of vacancy formation energy, can be 'low', 'medium', or 'high'. Default is 'medium'.
            The definition of the relax precision is the same as the relax_precision parameter in the keyword `relax_precision` in the `abacus_cal_band` function.
        
        The following parameters are only used when `property` is `md`:
        md_type (Literal['nve', 'nvt', 'npt', 'langevin']): The algorithm to integrate the equation of motion for molecular dynamics (MD).
            - nve: NVE ensemble with velocity Verlet algorithm.
            - nvt: NVT ensemble.
            - npt: Nose-Hoover style NPT ensemble.
            - langevin: NVT ensemble with Langevin thermostat.
        md_nstep (int): The total number of molecular dynamics steps.
        md_dt (float): The time step used in molecular dynamics calculations. THe unit in fs.
        md_tfirst (float): If set to larger than 0, initial velocity will be generated according to its value.
            If unset or smaller than 0, initial velocity will try to be read from STRU file.
        md_tlast (float): Only used in NVT/NPT simulations. If md_tlast is unset or less than zero, 
            md_tlast is set to md_tfirst. If md_tlast is set to be different from md_tfirst, 
            ABACUS will automatically change the temperature from md_tfirst to md_tlast
        md_thermostat (str): Specify the temperature control method used in NVT ensemble.
            - nhc: Nose-Hoover chain, see md_tfreq and md_tchain in detail.
            - anderson: Anderson thermostat, see md_nraise in detail.
            - berendsen: Berendsen thermostat, see md_nraise in detail.
            - rescaling: velocity Rescaling method 1, see md_tolerance in detail.
            - rescale_v: velocity Rescaling method 2, see md_nraise in detail.
        md_pmode (str): Specify the cell fluctuation mode in NPT ensemble based on the Nose-Hoover style non-Hamiltonian equations of motion.
            - iso: The three diagonal elements of the lattice are fluctuated isotropically.
            - aniso: The three diagonal elements of the lattice are fluctuated anisotropically.
            - tri: The lattice must be a lower-triangular matrix, and all six freedoms are fluctuated.
        md_pcouple (str): The coupled lattice vectors will scale proportionally in NPT ensemble based on the Nose-Hoover style non-Hamiltonian equations of motion.
            - none: Three lattice vectors scale independently.
            - xyz: Lattice vectors x, y, and z scale proportionally.
            - xy: Lattice vectors x and y scale proportionally.
            - xz: Lattice vectors x and z scale proportionally.
            - yz: Lattice vectors y and z scale proportionally.
        md_dumpfreq (int): The output frequency of OUT.${suffix}/MD_dump in molecular dynamics calculations. Generally the default value 1
            is OK. For very long ab-initio MD calculations, increasing md_dumpfreq can help reducing the size of MD_dump.
        md_seed (int): The random seed to initialize random numbers used in molecular dynamics calculations.
            - < 0: No srand() function is called.
            - >= 0: The function srand(md_seed) is called.
    
    Returns:
        A dictionary containing the calculation results of the property calculation.
    
    Raises:
        FileNotFoundError: If the structure file does not exist.
    """
    extra_input = {}
    if dft_functional in ['PBE', 'PBEsol', 'LDA', 'SCAN']:
        extra_input['dft_functional'] = dft_functional
    elif dft_functional in ['R2SCAN']:
        extra_input['dft_functional'] = 'MGGA_X_R2SCAN+MGGA_C_R2SCAN'
    elif dft_functional in [ 'HSE', 'PBE0']:
        print("Calculating with hybird functionals like HSE and PBE0 needs much longer time and much more meory than GGA functionals such as PBE.")
        extra_input['dft_functional'] = dft_functional
        os.environ['ABACUS_COMMAND'] = "OMP_NUM_THREADS=16 abacus" # Set to use OpenMP for hybrid functionals like HSE and PBE0
    else:
        print("DFT functional not supported now. Use dafault PBE functional.")
    
    abacus_prepare_outputs = abacus_prepare(stru_file=stru_file,
                                            stru_type=stru_type,
                                            lcao=lcao,
                                            nspin=nspin,
                                            #soc=soc,
                                            dftu=dftu,
                                            dftu_param=dftu_param,
                                            init_mag=init_mag,
                                            #afm=afm,
                                            extra_input=extra_input)
    
    print(abacus_prepare_outputs)
    
    abacus_inputs_dir = abacus_prepare_outputs['abacus_inputs_dir']

    if relax:
        relax_thresholds = get_relax_precision(relax_precision)
        
        if relax_cell is False: # For ABACUS LTSv3.10.0
            relax_method = 'bfgs_trad'
        else:
            relax_method = 'cg'
        
        max_steps = 100
        relax_outputs = abacus_do_relax(abacus_inputs_dir,
                                        force_thr_ev=relax_thresholds['force_thr_ev'],
                                        stress_thr_kbar=relax_thresholds['stress_thr'],
                                        max_steps=max_steps,
                                        relax_cell=relax_cell,
                                        relax_method=relax_method)
        
        if relax_outputs['result']['normal_end'] is False:
            raise ValueError('Relaxation calculation failed')
        elif relax_outputs['result']['relax_converge'] is False:
            return {"msg":f'Relaxation calculation did not converge in {max_steps} steps',
                    "final_stru": Path(relax_outputs['results']['new_abacus_inputs_dir']) / "STRU",
                    **relax_outputs["results"]}
        else:
            print("Relax calculation completed successfully.")
            abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']

    if property == 'bader_charge':
        outputs = abacus_badercharge_run(abacus_inputs_dir)
    elif property == 'elf':
        outputs = abacus_cal_elf(abacus_inputs_dir)
    elif property == 'band':
        outputs = abacus_cal_band(abacus_inputs_dir)
    elif property == 'dos':
        outputs = abacus_dos_run(abacus_inputs_dir)
    elif property == 'elastic_properties':
        outputs = abacus_cal_elastic(abacus_inputs_dir)
    elif property == 'eos':
        outputs = abacus_eos(abacus_inputs_dir)
    elif property == 'phonon_dispersion':
        outputs = abacus_phonon_dispersion(abacus_inputs_dir)
    elif property == 'md':
        outputs = abacus_run_md(abacus_inputs_dir, 
                                md_type,
                                md_nstep,
                                md_dt,
                                md_tfirst,
                                md_tlast,
                                md_thermostat,
                                md_pmode,
                                md_pcouple,
                                md_dumpfreq,
                                md_seed)
    elif property == 'work_function':
        outputs = abacus_cal_work_function(abacus_inputs_dir,
                                           vacuum_direction,
                                           dipole_correction)
    elif property == 'vacancy_formation_energy':
        outputs = abacus_cal_vacancy_formation_energy(abacus_inputs_dir,
                                                      vacancy_supercell,
                                                      vacancy_element,
                                                      vacancy_element_index,
                                                      vacancy_relax_precision)
    else:
        raise ValueError(f'Invalid property: {property}')
    
    return outputs

@mcp.tool()
def abacus_calculation_scf(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
) -> Dict[str, Any]:
    """
    Run ABACUS SCF calculation.

    Args:
        abacusjob (str): Path to the directory containing the ABACUS input files.
    Returns:
        A dictionary containing the path to output file of ABACUS calculation, and a dictionary containing whether the SCF calculation
        finished normally, the SCF is converged or not, the converged SCF energy and total time used.
    """
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    return _abacus_calculation_scf(abacus_inputs_dir)

@mcp.tool()
def abacus_do_relax(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
) -> Dict[str, Any]:
    """
    Perform relaxation calculations using ABACUS based on the provided input files. The results of the relaxation and 
    the new ABACUS input files containing final relaxed structure will be returned.
    """
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                             max_steps=max_steps,
                             relax_cell=relax_cell,
                             relax_precision=relax_precision,
                             fixed_axes=fixed_axes,
                             relax_method=relax_method)

    return relax_outputs

@mcp.tool()
def abacus_badercharge_run(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
) -> List[float]:
    """
    Run Bader charge calculation using ABACUS based on the provided input files. The results of the Bader charge calculation
    will be returned.
    """
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                             max_steps=max_steps,
                             relax_cell=relax_cell,
                             relax_precision=relax_precision,
                             fixed_axes=fixed_axes,
                             relax_method=relax_method)
    
    new_abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']

    badercharge_results = _abacus_badercharge_run(new_abacus_inputs_dir)

    return badercharge_results

@mcp.tool()
def abacus_dos_run(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    pdos_mode: Literal['species', 'species+shell', 'species+orbital'] = 'species+shell',
    dos_edelta_ev: float = 0.01,
    dos_sigma: float = 0.07,
    dos_scale: float = 0.01,
    dos_emin_ev: float = None,
    dos_emax_ev: float = None,
    dos_nche: int = None,
) -> Dict[str, Any]:
    """
    """
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']

    dos_results = _abacus_dos_run(abacus_inputs_dir,
                                  pdos_mode,
                                  dos_edelta_ev,
                                  dos_sigma,
                                  dos_scale,
                                  dos_emin_ev,
                                  dos_emax_ev,
                                  dos_nche)
    
    return dos_results

@mcp.tool()
def abacus_cal_band(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    mode: Literal["nscf", "pyatb", "auto"] = "auto",
    kpath: Union[List[str], List[List[str]]] = None,
    high_symm_points: Dict[str, List[float]] = None,
    energy_min: float = -10,
    energy_max: float = 10,
    insert_point_nums: int = 30
) -> Dict[str, Any]:
    """
    """
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    band_calculation_outputs = _abacus_cal_band(abacus_inputs_dir,
                                                mode,
                                                kpath,
                                                high_symm_points,
                                                energy_min,
                                                energy_max,
                                                insert_point_nums)
    
    return band_calculation_outputs

@mcp.tool()
def abacus_phonon_dispersion(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = True,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    supercell: Optional[List[int]] = None,
    displacement_stepsize: float = 0.01,
    temperature: Optional[float] = 298.15,
    min_supercell_length: float = 10.0,
    qpath: Optional[Union[List[str], List[List[str]]]] = None,
    high_symm_points: Optional[Dict[str, List[float]]] = None
) -> Dict[str, Any]:
    """
    Calculate phonon dispersion with finite-difference method using Phonopy with ABACUS as the calculator.
    """
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    phonon_outputs = _abacus_phonon_dispersion(abacus_inputs_dir,
                                               supercell,
                                               displacement_stepsize,
                                               temperature,
                                               min_supercell_length,
                                               qpath,
                                               high_symm_points)
    
    return phonon_outputs

@mcp.tool()
def abacus_cal_elastic(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = True,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    norm_strain: float = 0.01,
    shear_strain: float = 0.01,
    kspacing: float = 0.08,
    relax_force_thr_ev: float = 0.01
) -> Dict[str, Any]:
    """
    Calculate elastic properties using ABACUS.
    """
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    elactic_outputs = _abacus_cal_elastic(abacus_inputs_dir,
                                          norm_strain,
                                          shear_strain,
                                          kspacing,
                                          relax_force_thr_ev)
    
    return elactic_outputs

