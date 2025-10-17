from pathlib import Path
from typing import List, Dict, Any, Literal

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.work_function import abacus_cal_work_function as _abacus_cal_work_function

@mcp.tool()
def abacus_cal_work_function(
    abacus_inputs_dir: Path,
    vacuum_direction: Literal['x', 'y', 'z'] = 'z',
    dipole_correction: bool = False,
) -> Dict[str, Any]:
    """
    Calculate the electrostatic potential and work function using ABACUS.
    
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        vacuum_direction (Literal['x', 'y', 'z']): The direction of the vacuum.
        dipole_correction (bool): Whether to apply dipole correction along the vacuum direction.

    Returns:
        A dictionary containing:
        - elecstat_pot_work_function_work_path (Path): Path to the ABACUS job directory calculating electrostatic potential and work function.
        - elecstat_pot_file (Path): Path to the cube file containing the electrostatic potential.
        - averaged_elecstat_pot_plot (Path): Path to the plot of the averaged electrostatic potential.
        - work_function_results (list): A list of dictionary, where each dictionary contains 3 keys:
            - 'work_function': calculated work function
            - 'plateau_start_fractional': Fractional coordinate of start of the identified plateau in the given vacuum direction
            - 'plateau_end_fractional': Fractional coordinate of end of the identified plateau in the given vacuum direction
    """
    return _abacus_cal_work_function(abacus_inputs_dir, vacuum_direction, dipole_correction)
