from pathlib import Path
from typing import List, Dict, Any

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.vacancy import abacus_cal_vacancy_formation_energy as _abacus_cal_vacancy_formation_energy

@mcp.tool()
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
        vacancy_element (str): Element to be removed. Default is None, which means the first type of element in the structure.
        vacancy_element_index (int): Index of the vacancy element. Defaults to 1. The index is in the original structure.
    Returns:
        A dictionary containing:
        - "vacancy_formation_energy": Calculated vacancy formation energy.
        - "supercell_jobpath": Path to the supercell calculation job directory.
        - "defect_supercell_jobpath": Path to the defect supercell calculation job directory.
        - "vacancy_element_crys_jobpath": Path to the most stable crystal structure calculation job directory.
    """
    return _abacus_cal_vacancy_formation_energy(
        abacus_inputs_dir,
        supercell,
        vacancy_element,
        vacancy_element_index
    )
