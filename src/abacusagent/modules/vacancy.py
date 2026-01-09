from pathlib import Path
from typing import List, Dict, Any, Literal

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.vacancy import abacus_cal_vacancy_formation_energy as _abacus_cal_vacancy_formation_energy

@mcp.tool()
def abacus_cal_vacancy_formation_energy(
    abacus_inputs_dir: Path,
    supercell: List[int],
    vacancy_index: int,
    relax_precision: Literal['low', 'medium', 'high'] = 'low',
) -> Dict[str, Any]:
    """
    Calculate vacancy formation energy. Currenly only non-charged vacancy of limited elements are suppoted. 
    Supported elements include: Li, Be, Na, Mg, Al, Si, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, 
    Ge, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, 
    Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb.
    The most stable crystal structure are used.

    Args:
        abacus_inputs_dir (Path): Path to the directory containing the ABACUS inputs.
        supercell (List[int]): Supercell matrix. Defaults to [1, 1, 1], which means no supercell.
        vacancy_index (int): Index of the vacancy element. The index starts from 1 and is in the original structure. Defaults to 1.
        relax_precision (Literal['low', 'medium', 'high']): Precision of the relax calculation. The unit of `force_thr_ev` is eV/Angstrom, and the unit of `stress_thr` is kbar.
        - 'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr=5.0.
        - 'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr=1.0.
        - 'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr=0.5.
    Returns:
        A dictionary containing:
        - "vacancy_formation_energy": Calculated vacancy formation energy.
        - "work_path": Path to the work path of vacancy formation energy calculation.
        - "original_stru_job_relax_converge": If the relax calculation of original structure is converged.
        - "defect_supercell_job_relax_converge": If the defect supercell relax calculation is converged.
    """
    return _abacus_cal_vacancy_formation_energy(
        abacus_inputs_dir,
        supercell,
        vacancy_index,
        relax_precision
    )
