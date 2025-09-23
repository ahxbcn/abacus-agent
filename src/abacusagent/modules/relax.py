import os
import json
from pathlib import Path
from typing import Literal, Optional,  Dict, Any
from abacustest.lib_prepare.abacus import ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.relax import abacus_do_relax as _abacus_do_relax

@mcp.tool()
def abacus_do_relax(
    abacus_inputs_dir: Path,
    force_thr_ev: Optional[float] = None,
    stress_thr_kbar: Optional[float] = None,
    max_steps: Optional[int] = None,
    relax_cell: Optional[bool] = None,
    fixed_axes: Optional[Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"]] = None,
    relax_method: Optional[Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"]] = None,
    relax_new: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Perform relaxation calculations using ABACUS based on the provided input files. The results of the relaxation and 
    the new ABACUS input files containing final relaxed structure will be returned.
    
    Args:
        abacus_inputs_dir: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
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

    Returns:
        A dictionary containing:
        - job_path: The job path of the relaxation calculation.
        - new_abacus_inputs_dir: The path to the new ABACUS input files using relaxed structure in STRU file from the relaxation results.
                                  Property calculation should be performed using this new ABACUS input files.
        - result: The result of the relaxation calculation with a dictionary containing:
            - normal_end: Whether the relaxation calculation ended normally.
            - relax_steps: The number of relaxation steps taken.
            - largest_gradient: The largest force gradient during the relaxation.
            - relax_converge: Whether the relaxation converged.
            - energies: The energies at each step of the relaxation.
    
    For example:
        # only relax the atomic positions
        >>> abacus_do_relax(
                abacus_inputs_dir="/path/to/abacus/inputs",
                force_thr_ev=0.01,
                max_steps=100,
                relax_cell=False)
        # relax the cell parameters and atomic positions
        >>> abacus_do_relax(
                abacus_inputs_dir="/path/to/abacus/inputs",
                force_thr_ev=0.01,
                stress_thr_kbar=1.0,
                max_steps=100,
                relax_cell=True)
        # relax the cell parameters and atomic positions with fixed volume
        >>> abacus_do_relax(
                abacus_inputs_dir="/path/to/abacus/inputs",
                force_thr_ev=0.01,
                stress_thr_kbar=1.0,
                max_steps=100,
                relax_cell=True,
                fixed_axes="volume") 
        
        When the relaxation is not converged, please try to use other relaxation methods.
    """
    return _abacus_do_relax(abacus_inputs_dir,
                           force_thr_ev=force_thr_ev,
                           stress_thr_kbar=stress_thr_kbar,
                           max_steps=max_steps,
                           relax_cell=relax_cell,
                           fixed_axes=fixed_axes,
                           relax_method=relax_method,
                           relax_new=relax_new
    )
    