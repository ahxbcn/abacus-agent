from pathlib import Path
from typing import List, Dict, Any, Literal

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.work_function import abacus_cal_work_function as _abacus_cal_work_function

@mcp.tool()
def abacus_cal_work_function(
    abacus_inputs_dir: Path,
    vacuum_direction: Literal['x', 'y', 'z'] = 'z',
) -> Dict[str, Any]:
    """
    Calculate the electrostatic potential and work function using ABACUS.
    
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        vacuum_direction (Literal['x', 'y', 'z']): The direction of the vacuum.

    Returns:
        A dictionary containing:
        - elecstat_pot_work_function_work_path (Path): Path to the ABACUS job directory calculating electrostatic potential and work function.
        - elecstat_pot_file (Path): Path to the cube file containing the electrostatic potential.
        - averaged_elecstat_pot_plot (Path): Path to the plot of the averaged electrostatic potential.
        - work_function (float): The calculated work function in eV.
    """
    return _abacus_cal_work_function(abacus_inputs_dir, vacuum_direction)
