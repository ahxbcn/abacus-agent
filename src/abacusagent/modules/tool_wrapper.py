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

@mcp.tool()
def run_abacus_calculation(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    property: Literal['bader_charge', 'elf', 'band', 'dos', 'elastic_properties', 'eos', 'phonon_dispersion',] = 'dos',
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    afm: bool = False,
):
    """
    Calculate properties using ABACUS.

    Args:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'Low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'Medium' means the relax calculation will be done with force_thr_ev=0.02 and stress_thr_kbar=1.0.
            'High' means the relax calculation will be done with force_thr_ev=0.002 and stress_thr_kbar=0.1.
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
        extra_property_params: Extra parameters for the property calculation.
    
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
        extra_input['dft_functional'] = dft_functional
        os.environ['ABACUS_COMMAND'] = "OMP_NUM_THREADS=16 abacus" # Set to use OpenMP for hybrid functionals like HSE and PBE0
    else:
        print("DFT functional not supported now. Use dafault PBE functional.")
    
    abacus_prepare_outputs = abacus_prepare(stru_file=stru_file,
                                            stru_type=stru_type,
                                            lcao=lcao,
                                            nspin=nspin,
                                            soc=soc,
                                            dftu=dftu,
                                            dftu_param=dftu_param,
                                            init_mag=init_mag,
                                            afm=afm,
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
        
        if relax_cell is False: # For ABACUS LTSv3.10
            relax_method = 'bfgs_trad'
        else:
            relax_method = 'cg'

        relax_outputs = abacus_do_relax(abacus_inputs_dir,
                                        force_thr_ev=force_thr_ev,
                                        stress_thr_kbar=stress_thr_kbar,
                                        max_steps=100,
                                        relax_cell=relax_cell,
                                        relax_method=relax_method)
        
        if relax_outputs['result']['normal_end'] is False:
            raise ValueError('Relaxation calculation failed')
        elif relax_outputs['result']['relax_converge'] is False:
            raise ValueError('Relaxation calculation did not converge')
        else:
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
    else:
        raise ValueError(f'Invalid property: {property}')
    
    return outputs
    