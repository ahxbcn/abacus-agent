from pathlib import Path
import os
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.abacus import abacus_prepare
from abacusagent.modules.submodules.cube import abacus_cal_elf
from abacusagent.modules.submodules.band import abacus_cal_band
from abacusagent.modules.submodules.bader import abacus_badercharge_run
from abacusagent.modules.submodules.dos import abacus_dos_run
from abacusagent.modules.submodules.phonon import abacus_phonon_dispersion
from abacusagent.modules.submodules.elastic import abacus_cal_elastic
from abacusagent.modules.submodules.eos import abacus_eos
from abacusagent.modules.submodules.relax import abacus_do_relax
from abacusagent.modules.submodules.md import abacus_run_md

@mcp.tool()
def run_abacus_calculation(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    property: Literal['bader_charge', 'elf', 'band', 'dos', 'elastic_properties', 'eos', 'phonon_dispersion', 'md'] = 'bader_charge',
    lcao: bool = True,
    nspin: Literal[1, 2, 4] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
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
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'Low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'Medium' means the relax calculation will be done with force_thr_ev=0.02 and stress_thr_kbar=1.0.
            'High' means the relax calculation will be done with force_thr_ev=0.002 and stress_thr_kbar=0.1.
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
        if relax_precision == 'low':
            force_thr_ev, stress_thr_kbar = 0.05, 5
        elif relax_precision == 'medium':
            force_thr_ev, stress_thr_kbar = 0.02, 1.0
        elif relax_precision == 'high':
            force_thr_ev, stress_thr_kbar = 0.002, 0.1
        else:
            raise ValueError(f'Invalid relax_precision: {relax_precision}')
        
        if relax_cell is False: # For ABACUS LTSv3.10.0
            relax_method = 'bfgs_trad'
        else:
            relax_method = 'cg'
        
        max_steps = 100
        relax_outputs = abacus_do_relax(abacus_inputs_dir,
                                        force_thr_ev=force_thr_ev,
                                        stress_thr_kbar=stress_thr_kbar,
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
    else:
        raise ValueError(f'Invalid property: {property}')
    
    return outputs
    