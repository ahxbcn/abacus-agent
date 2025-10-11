from typing import Dict, List, Optional, Any, Literal
from pathlib import Path

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.phonon import abacus_phonon_dispersion as _abacus_phonon_dispersion

@mcp.tool()
def abacus_phonon_dispersion(
    abacus_inputs_dir: Path,
    supercell: Optional[List[int]] = None,
    displacement_stepsize: float = 0.01,
    temperature: Optional[float] = 298.15,
    min_supercell_length: float = 10.0,
):
    """
    Calculate phonon dispersion with finite-difference method using Phonopy with ABACUS as the calculator. 
    This tool function is usually followed by a cell-relax calculation (`calculation` is set to `cell-relax`). 
    Args:
        abacus_inputs_dir (Path): Path to the directory containing ABACUS input files.
        supercell (List[int], optional): Supercell matrix for phonon calculations. If default value None are used,
            the supercell matrix will be determined by how large a supercell can have a length of lattice vector
            along all 3 directions larger than 10.0 Angstrom.
        displacement_stepsize (float, optional): Displacement step size for finite difference. Defaults to 0.01 Angstrom.
        temperature (float, optional): Temperature in Kelvin for thermal properties. Defaults to 298.15. Units in Kelvin.
        min_supercell_length (float): If supercell is not provided, the generated supercell will have a length of lattice vector
            along all 3 directions larger than min_supercell_length. Defaults to 10.0 Angstrom. Units in Angstrom.
    Returns:
        A dictionary containing:
            - phonon_work_path: Path to the directory containing phonon calculation results.
            - band_plot: Path to the phonon dispersion plot.
            - dos_plot: Path to the phonon density of states plot.
            - entropy: Entropy at the specified temperature.
            - free_energy: Free energy at the specified temperature.
            - heat_capacity: Heat capacity at the specified temperature.
            - max_frequency_THz: Maximum phonon frequency in THz.
            - max_frequency_K: Maximum phonon frequency in Kelvin.
    """
    return _abacus_phonon_dispersion(
        abacus_inputs_dir,
        supercell,
        displacement_stepsize,
        temperature,
        min_supercell_length
    )
