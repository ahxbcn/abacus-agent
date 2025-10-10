import os
from pathlib import Path
from typing import List, Dict, Any
from itertools import groupby

from ase.build import bulk

from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.modules.util.comm import generate_work_path, link_abacusjob, run_abacus, collect_metrics

# From Introduction to Solid State Physics, 8th edition, by Charles Kittel
ELEMENT_CRYSTAL_STRUCTURES = {
    "Li": {"crystal": "bcc", "a": 3.51},
    "Be": {"crystal": "hcp", "a": 2.27, "c": 3.59},
    "Na": {"crystal": "bcc", "a": 4.23},
    "Mg": {"crystal": "hcp", "a": 3.21, "c": 5.21},
    "Al": {"crystal": "fcc", "a": 4.05},
    "Si": {"crystal": "diamond", "a": 5.43},
    "K": {"crystal": "bcc", "a": 5.23},
    "Ca": {"crystal": "fcc", "a": 5.58},
    "Sc": {"crystal": "hcp", "a": 3.31, "c": 5.27},
    "Ti": {"crystal": "hcp", "a": 2.95, "c": 4.68},
    "V": {"crystal": "bcc", "a": 3.03},
    "Cr": {"crystal": "bcc", "a": 2.88},
    "Mn": {"crystal": "bcc", "a": 2.91}, # To be checked
    "Fe": {"crystal": "bcc", "a": 2.87},
    "Co": {"crystal": "hcp", "a": 2.51, "c": 4.07},
    "Ni": {"crystal": "fcc", "a": 3.52},
    "Cu": {"crystal": "fcc", "a": 3.61},
    "Zn": {"crystal": "hcp", "a": 2.66, "c": 4.95},
    "Ga": {"crystal": "fcc", "a": 4.50}, # To be checked
    "Ge": {"crystal": "diamond", "a": 5.69},
    "Rb": {"crystal": "bcc", "a": 5.58},
    "Sr": {"crystal": "fcc", "a": 6.08},
    "Y": {"crystal": "hcp", "a": 3.65, "c": 5.73},
    "Zr": {"crystal": "hcp", "a": 3.23, "c": 5.15},
    "Nb": {"crystal": "bcc", "a": 3.30},
    "Mo": {"crystal": "bcc", "a": 3.15},
    "Tc": {"crystal": "hcp", "a": 2.74, "c": 4.44},
    "Ru": {"crystal": "hcp", "a": 2.71, "c": 4.28},
    "Rh": {"crystal": "fcc", "a": 3.80},
    "Pd": {"crystal": "fcc", "a": 3.89},
    "Ag": {"crystal": "fcc", "a": 4.09},
    "Cd": {"crystal": "hcp", "a": 2.98, "c": 5.62},
    "In": {"crystal": "bcc", "a": 3.25}, # To be checked
    "Sn": {"crystal": "diamond", "a": 6.49},
    "Cs": {"crystal": "bcc", "a": 6.05},
    "Ba": {"crystal": "bcc", "a": 5.02},
    "La": {"crystal": "hcp", "a": 3.75, "c": 5.75}, # To be checked
    "Ce": {"crystal": "fcc", "a": 5.16}, 
    "Pr": {"crystal": "hcp", "a": 3.65, "c": 5.75}, # To be checked
    "Nd": {"crystal": "hcp", "a": 3.65, "c": 5.73}, # To be checked
    "Pm": {"crystal": "hcp", "a": 3.65, "c": 5.73}, # To be checked
    "Sm": {"crystal": "hcp", "a": 3.65, "c": 5.73}, # To be checked 
    "Eu": {"crystal": "bcc", "a": 4.58},
    "Gd": {"crystal": "hcp", "a": 3.63, "c": 5.78},
    "Tb": {"crystal": "hcp", "a": 3.60, "c": 5.70},
    "Dy": {"crystal": "hcp", "a": 3.59, "c": 5.65},
    "Ho": {"crystal": "hcp", "a": 3.58, "c": 5.62},
    "Er": {"crystal": "hcp", "a": 3.56, "c": 5.59},
    "Tm": {"crystal": "hcp", "a": 3.54, "c": 5.56},
    "Yb": {"crystal": "fcc", "a": 5.48},
    "Lu": {"crystal": "hcp", "a": 3.50, "c": 5.55},
    "Hf": {"crystal": "hcp", "a": 3.19, "c": 5.05},
    "Ta": {"crystal": "bcc", "a": 3.30},
    "W": {"crystal": "bcc", "a": 3.16},
    "Re": {"crystal": "hcp", "a": 2.76, "c": 4.46},
    "Os": {"crystal": "hcp", "a": 2.74, "c": 4.32},
    "Ir": {"crystal": "fcc", "a": 3.84},
    "Pt": {"crystal": "fcc", "a": 3.92},
    "Au": {"crystal": "fcc", "a": 4.08},
    "Hg": {"crystal": "hcp", "a": 2.99, "c": 5.01}, # To be checked
    "Tl": {"crystal": "hcp", "a": 3.46, "c": 5.52},
    "Pb": {"crystal": "fcc", "a": 4.95},
}

MAGNETIC_CRYSTALS = {"Fe", "Ni", "Co", "Cr", "Mn"}

def build_most_stable_elementary_crys_stru(element: str, pp: str, orb: str) -> AbacusStru:
    """
    Build the most stable crystal structure of an element.
    Args:
        element (str): The chemical symbol of the element.
        pp (dict): Pseudopotential information from the original structure.
        orb (dict): Orbital information from the original structure.
    Returns:
        AbacusStru: The most stable crystal structure of the element.
    Raises:
        ValueError: If the element is not supported.
    """
    if element not in ELEMENT_CRYSTAL_STRUCTURES:
        raise ValueError(f"Element {element} not supported.")
    else:
        crystal_structure = ELEMENT_CRYSTAL_STRUCTURES[element]
        if crystal_structure['crystal'] == 'bcc':
            stru = bulk(element, 'bcc', a=crystal_structure['a'])
        elif crystal_structure['crystal'] == 'fcc':
            stru = bulk(element, 'fcc', a=crystal_structure['a'])
        elif crystal_structure['crystal'] == 'hcp':
            stru = bulk(element, 'hcp', a=crystal_structure['a'], c=crystal_structure['c'])
        elif crystal_structure['crystal'] == 'diamond':
            stru = bulk(element, 'diamond', a=crystal_structure['a'])
        else:
            raise ValueError(f"Crystal structure {crystal_structure['crystal']} not supported.")
    
    structure_file = f"{element}_{crystal_structure['crystal']}.STRU"
    stru.write(structure_file, format="abacus")

    stru_abacus = AbacusStru.ReadStru(structure_file)
    stru_abacus.set_pp([pp])
    stru_abacus.set_orb([orb])
    os.unlink(structure_file)
    return stru_abacus
    
def abacus_cal_vacancy_formation_energy(
    abacus_inputs_dir: Path,
    supercell: List[int] = [1, 1, 1],
    vacancy_element: str = None,
    vacancy_element_index: int = 1,
) -> Dict[str, Any]:
    """
    Calculate vacancy formation energy. Currenly only non-charged vacancy of limited elements are suppoted. 
    Supported elements include: Li, Be, Na, Mg, Al, Si, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, 
    Ge, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, 
    Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb.
    The most stable crystal structure are used.

    Args:
        abacus_inputs_dir (Path): Path to the directory containing the ABACUS inputs.
        supercell_matrix (List[int]): Supercell matrix. Defaults to [1, 1, 1], which means no supercell.
        vacancy_element (str): Element to be removed. Defaults to "Fe".
        vacancy_element_index (int): Index of the vacancy element. The index starts from 1 and is in the original structure. Defaults to 1.
    Returns:
        A dictionary containing:
        - "vacancy_formation_energy": Calculated vacancy formation energy.
        - "supercell_jobpath": Path to the supercell calculation job directory.
        - "defect_supercell_jobpath": Path to the defect supercell calculation job directory.
        - "vacancy_element_crys_jobpath": Path to the most stable crystal structure calculation job directory.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")

        work_path = Path(generate_work_path()).absolute()
        original_inputs_dir = os.path.join(work_path, "original_inputs")
        link_abacusjob(src=abacus_inputs_dir, dst=original_inputs_dir, copy_files=['INPUT', 'STRU'])
        input_params = ReadInput(os.path.join(original_inputs_dir, "INPUT"))
        original_stru = AbacusStru.ReadStru(os.path.join(original_inputs_dir, input_params.get("stru_file", "STRU")))

        supercell_stru = original_stru.supercell(supercell)
        vacancy_element_index -= 1
        defect_supercell_stru = supercell_stru.delete_atom(vacancy_element, vacancy_element_index)

        input_params['calculation'] = 'cell-relax'
        input_params['relax_method'] = 'cg'
        input_params['force_thr_ev'] = 0.02
        input_params['stress_thr'] = 0.5

        # Prepare ABACUS input files for supercell structure, defect supercell structure, and most stable crystal structure of the vacancy element
        supercell_jobpath = os.path.join(work_path, "supercell_cell_relax")
        link_abacusjob(src=original_inputs_dir, dst=supercell_jobpath, copy_files=['INPUT', 'STRU', "abacus.log"])
        WriteInput(input_params, os.path.join(supercell_jobpath, "INPUT"))
        supercell_stru.write(os.path.join(supercell_jobpath, input_params.get("stru_file", "STRU")))

        defect_supercell_jobpath = os.path.join(work_path, "defect_supercell_cell_relax")
        link_abacusjob(src=original_inputs_dir, dst=defect_supercell_jobpath, copy_files=['INPUT', 'STRU', "abacus.log"])
        WriteInput(input_params, os.path.join(defect_supercell_jobpath, "INPUT"), )
        defect_supercell_stru.write(os.path.join(defect_supercell_jobpath, input_params.get("stru_file", "STRU")))

        element_type_index = [key for key, group in groupby(original_stru.get_element(number=False))].index(vacancy_element)
        vacancy_element_pp = original_stru.get_pp()[element_type_index]
        vacancy_element_orb = original_stru.get_orb()[element_type_index]
        vacancy_element_crys_stru = build_most_stable_elementary_crys_stru(vacancy_element, vacancy_element_pp, vacancy_element_orb)
        vacancy_element_crys_jobpath = os.path.join(work_path, f"{vacancy_element}_crys_cell_relax")
        link_abacusjob(src=original_inputs_dir, dst=vacancy_element_crys_jobpath, copy_files=['INPUT', 'STRU', "abacus.log"])
        if vacancy_element in MAGNETIC_CRYSTALS:
            input_params['nspin'] = 2
        WriteInput(input_params, os.path.join(vacancy_element_crys_jobpath, "INPUT"))
        vacancy_element_crys_stru.write(os.path.join(vacancy_element_crys_jobpath, input_params.get("stru_file", "STRU")))

        run_abacus([supercell_jobpath,
                    defect_supercell_jobpath,
                    vacancy_element_crys_jobpath])

        # Calculate the vacancy formation energy
        supercell_energy = collect_metrics(supercell_jobpath, metrics_names=['energy'])['energy']
        defect_supercell_energy = collect_metrics(defect_supercell_jobpath, metrics_names=['energy'])['energy']
        vacancy_element_crys_energy = collect_metrics(vacancy_element_crys_jobpath, metrics_names=['energy'])['energy']

        vac_formation_energy = (defect_supercell_energy + vacancy_element_crys_energy / vacancy_element_crys_stru.get_natoms()) - supercell_energy

        return {'vac_formation_energy': vac_formation_energy,
                'supercell_jobpath': Path(supercell_jobpath).absolute(),
                'defect_supercell_jobpath': Path(defect_supercell_jobpath).absolute(),
                'vacancy_element_crys_jobpath': Path(vacancy_element_crys_jobpath).absolute()}
    except Exception as e:
        raise RuntimeError(f"Error in abacus_cal_vacancy_formation_energy: {e}")
