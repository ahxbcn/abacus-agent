import os
from pathlib import Path
from typing import Literal, Tuple

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.structure_editor import build_slab as _build_slab

@mcp.tool()
def build_slab(stru_file: Path,
               stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
               miller_indices: Tuple[int, int, int] = (1, 1, 1),
               layers: int = 3,
               supercell_2d: Tuple[int, int] = (1, 1),
               vacuum: float = 10.0,
               vacuum_direction: Literal['x', 'y', 'z'] = 'y'
):
    """
    Build slab from given structure file.

    Args:
        stru_file (Path): Path to structure file.
        stru_type (Literal["cif", "poscar", "abacus/stru"]): Type of structure file.
        miller_indices (Tuple[int, int, int]): Miller indices of the surface.
        layers (int, optional): Number of layers of the surface. Note that the layers is number of equivalent layers, not number of layers of atoms. Defaults to 3.
        vacuum (float, optional): Vacuum space between the surface and the bulk structure. Units in Angstrom. Defaults to 10.0 Angstrom.
    Returns:
        A dictionary containing the path to the surface structure file.
        Keys:
            - surface_stru_file: Path to the surface structure file.
    Raises:
        ValueError: If stru_type is not supported.
    """
    return _build_slab(stru_file, stru_type, miller_indices, layers, supercell_2d, vacuum, vacuum_direction)
