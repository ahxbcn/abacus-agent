from typing import Dict, List, Optional, Any, Literal, Union
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
    qpath: Optional[Union[List[str], List[List[str]]]] = None,
    high_symm_points: Optional[Dict[str, List[float]]] = None
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
        qpath (Tuple[List[str], List[List[str]]]): 
            A list of name of high symmetry points in the phonon dispersion path. Non-continuous line of high symmetry points are stored as seperate lists.
            For example, ['G', 'M', 'K', 'G'] and [['G', 'X', 'P', 'N', 'M', 'S'], ['S_0', 'G', R']] are both acceptable inputs.
            Default is None. If None, will use automatically generated q-point path.
            `kpath` must be used with `high_symm_points` to take effect.
        high_symm_points: A dictionary containing high symmetry points and their coordinates in the band path. All points in `qpath` should be included.
            For example, {'G': [0, 0, 0], 'M': [0.5, 0.0, 0.0], 'K': [0.33333333, 0.33333333, 0.0], 'G': [0, 0, 0]}.
            Default is None. If None, will use automatically generated high symmetry points.
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
        min_supercell_length,
        qpath,
        high_symm_points
    )
