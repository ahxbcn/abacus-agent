from pathlib import Path
from typing import Dict, Any

from abacusagent.init_mcp import mcp

@mcp.tool()
def get_high_symm_points_from_abacus_inputs_dir(abacusjob_dir: Path) -> Dict[str, Any]:
    """
    Get high symmetry points and kpath for STRU file in ABACUS inputs directory.
    Args:
        abacusjob_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    Returns:
        A dictionary containing high symmetry points and suggested kpath for STRU file in ABACUS inputs directory. The most important keys are:
        - path (List[List[str]]): Suggested path for the given structure.
        - point_coords: Coordinates of high symmetry points in reciprocal space.
    """
    from abacusagent.modules.util.symmetry import get_high_symm_points_from_abacus_inputs_dir as _get_high_symm_points_from_abacus_inputs_dir

    return _get_high_symm_points_from_abacus_inputs_dir(abacusjob_dir)
