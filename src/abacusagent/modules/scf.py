from pathlib import Path
from typing import Dict, Any

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.scf import abacus_calculation_scf as _abacus_calculation_scf

@mcp.tool()
def abacus_calculation_scf(
    abacus_inputs_dir: Path,
) -> Dict[str, Any]:
    """
    Run ABACUS SCF calculation.

    Args:
        abacusjob (str): Path to the directory containing the ABACUS input files.
    Returns:
        A dictionary containing the path to output file of ABACUS calculation, and a dictionary containing whether the SCF calculation
        finished normally, the SCF is converged or not, the converged SCF energy and total time used.
    """
    return _abacus_calculation_scf(abacus_inputs_dir)
