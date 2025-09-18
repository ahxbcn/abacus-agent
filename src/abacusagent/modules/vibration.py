from typing import Dict, List, Optional, Any, Literal
from pathlib import Path

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.vibration import abacus_vibration_analysis as _abacus_vibration_analysis

@mcp.tool()
def abacus_vibration_analysis(abacus_inputs_dir: Path,
                              selected_atoms: Optional[List[int]] = None,
                              stepsize: float = 0.01,
                              nfree: Literal[2, 4] = 2,
                              temperature: Optional[float] = 298.15):
    """
    Performing vibrational analysis using finite displacement method.
    This tool function is usually followed by a relax calculation (`calculation` is set to `relax`).
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files directory.
        selected_atoms (Optional[List[int]]): Indices of atoms included in the vibrational analysis. If this
            parameter are not given, all atoms in the structure will be included.
        stepsize (float): Step size to displace cartesian coordinates of atoms during the vibrational analysis.
            Units in Angstrom. The default value (0.01 Angstrom) is generally OK.
        nfree (int): Number of force calculations performed for each cartesian coordinate components of each 
            included atom. Allowed values are 2 and 4, where 2 represents calculating matrix element of force constant
            matrix using 3-point center difference and need 2 SCF calculations, and 4 means using 5-point center
            difference and need 4 SCF calculations. Generally `nfree=2` is accurate enough.
        temperature (float): Temperature used to calculate thermodynamic quantities. Units in Kelvin.
    Returns:
        A dictionary containing the following keys:
        - 'frequencies': List of real frequencies from vibrational analysis. Imaginary frequencies are represented by negative 
            values. Units in cm^{-1}.
        - 'zero_point_energy': Zero-point energy summed over all modes. Units in eV.
        - 'vib_entropy': Vibrational entropy using harmonic approximation. Units in eV/K.
        - 'vib_free_energy': Vibrational Helmholtz free energy using harmonic approximation. Units in eV.
        - 'vib_analysis_work_path': Path to directory performing vibrational analysis. Containing animation of normal modes 
            with non-zero frequency in ASE traj format and `vib` directory containing collected forces.
    """
    return _abacus_vibration_analysis(abacus_inputs_dir, selected_atoms, stepsize, nfree, temperature)
