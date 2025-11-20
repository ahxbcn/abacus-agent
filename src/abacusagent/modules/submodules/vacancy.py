import os
from pathlib import Path
from typing import List, Dict, Any, Literal

from abacustest.lib_model.comm import check_abacus_inputs
from abacustest.lib_model.model_017_vacancy import prepare_vacancy_jobs, postprocess_vacancy

from abacusagent.modules.util.comm import generate_work_path, link_abacusjob, run_abacus, get_relax_precision

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
        - "supercell_job_relax_converge": If the supercell relax calculation is converged.
        - "defect_supercell_job_relax_converge": If the defect supercell relax calculation is converged.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")

        work_path = Path(generate_work_path()).absolute()
        original_inputs_dir = os.path.join(work_path, "original_inputs")
        ref_dir = os.path.join(work_path, "ref_element")
        link_abacusjob(src=abacus_inputs_dir, dst=original_inputs_dir, copy_files=['INPUT', 'STRU'])

        relax_precision_values = get_relax_precision(relax_precision)

        job_dirs = prepare_vacancy_jobs([original_inputs_dir],
                                        supercell=supercell,
                                        original_vacancy_indices=[vacancy_index],
                                        cal_reference=True,
                                        ref_dir=ref_dir,
                                        max_step=100,
                                        force_thr_ev=relax_precision_values['force_thr_ev'],
                                        stress_thr_kbar=relax_precision_values['stress_thr'])
        
        run_abacus(job_dirs)

        results, ref_atom_energy = postprocess_vacancy(jobs=[original_inputs_dir],
                                                       ref_dir=ref_dir)
        
        full_return_dict = results[list(results.keys())[0]]
        returned_keys = ['vac_formation_energy', 'supercell_job_relax_converge', 'defect_supercell_job_relax_converge']
        returned_dict = {k: full_return_dict[k] for k in returned_keys}
        returned_dict['work_path'] = Path(work_path).absolute()
        return returned_dict
    except Exception as e:
        raise RuntimeError(f"Error in abacus_cal_vacancy_formation_energy: {e}")
