"""
Calculating elastic constants using ABACUS.
"""
from typing import Dict
from pathlib import Path

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.elastic import abacus_cal_elastic as _abacus_cal_elastic

@mcp.tool()
def abacus_cal_elastic(
    abacus_inputs_dir: Path,
    norm_strain: float = 0.01,
    shear_strain: float = 0.01,
    kspacing: float = 0.08,
    relax_force_thr_ev: float = 0.01
) -> Dict[str, float]:
    """
    Calculate various elastic constants for a given structure using ABACUS. 
    Args:
        abacus_inputs_dir (str): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        norm_strain (float): Normal strain to calculate elastic constants, default is 0.01.
        shear_strain (float): Shear strain to calculate elastic constants, default is 0.01.
        kspacing (float): K-point spacing for ABACUS calculation, default is 0.08. Units in Bohr^{-1}.
        relax_force_thr_ev (float): Threshold for force convergence of the relax calculation for each deformed structure, default is 0.02. Units in eV/Angstrom.

    Returns:
        A dictionary containing the following keys:
        - elastic_cal_dir (Path): Work path of running abacus_cal_elastic. 
        - elastic_constants (np.array in (6,6) dimension): Calculated elastic constants in Voigt notation. Units in GPa.
        - bulk_modulus (float): Calculated bulk modulus in GPa.
        - shear_modulus (float): Calculated shear modulus in GPa.
        - young_modulus (float): Calculated Young's modulus in GPa.
        - poisson_ratio (float): Calculated Poisson's ratio.
    Raises:
        RuntimeError: If ABACUS calculation when calculating stress for input structure or deformed structures fails.
    """
    return _abacus_cal_elastic(abacus_inputs_dir, norm_strain, shear_strain, kspacing, relax_force_thr_ev)
