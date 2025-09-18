from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.eos import abacus_cal_eos as _abacus_cal_eos

@mcp.tool()
def abacus_eos(
    abacus_inputs_dir: Path,
    stru_scale_number: int = 3,
    stru_scale_type: Literal['percentage', 'angstrom'] = 'percentage',
    scale_stepsize: float = 0.02
):
    """
    Use Birch-Murnaghan equation of state (EOS) to calculate the EOS data. The shape of fitted crystal is limited to cubic now.

    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        stru_scale_number (int): Number of structures to generate for EOS calculation.
        stru_scale_type (Literal['percentage', 'angstrom']): Type of scaling for structures.
        scale_stepsize (float): Step size for scaling.
            - 'percentage' means percentage of the original cell size. Default is 0.02, which means 2% of the original cell size.
            - 'angstrom' means absolute angstrom value. The typical stepsize 0.1 angstrom is recommended for most system.

    Returns:
        Dict[str, Any]: A dictionary containing EOS calculation results:
            - "eos_work_path" (Path): Working directory for the EOS calculation.
            - "new_abacus_inputs_dir" (Path): ABACUS input files directory containing the lowest energy structure using the fitted EOS.
            - "eos_fig_path" (Path): Path to the EOS fitting plot (energy vs. volume).
            - "E0" (float): Minimum energy (in eV) from the EOS fit.
            - "V0" (float): Equilibrium volume (in Å³) corresponding to E0.
            - "B0" (float): Bulk modulus (in GPa) at equilibrium volume.
            - "B0_deriv" (float): Pressure derivative of the bulk modulus.
    """
    return _abacus_cal_eos(abacus_inputs_dir, stru_scale_number, stru_scale_type, scale_stepsize)
