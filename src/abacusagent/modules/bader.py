from pathlib import Path
from typing import List, Dict, Optional, Any

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.bader import abacus_badercharge_run as _abacus_badercharge_run
from abacusagent.modules.submodules.bader import calculate_bader_charge_from_cube as _calculate_bader_charge_from_cube

@mcp.tool() # make it visible to the MCP server
def abacus_badercharge_run(
    abacus_inputs_dir: Path
) -> List[float]:
    """
    Calculate Bader charges for a given ABACUS input file directory, with ABACUS as
    the dft software to calculate the charge density, and then postprocess
    the charge density with the cube manipulator and Bader analysis.
    
    Parameters:
    abacus_inputs_dir (str): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
    
    Returns:
    dict: A dictionary containing: 
        - bader_charges: List of Bader charge for each atom.
        - atom_labels: Labels of atoms in the structure.
        - abacus_workpath: Absolute path to the ABACUS work directory.
        - badercharge_run_workpath: Absolute path to the Bader analysis work directory.
    """
    return _abacus_badercharge_run(abacus_inputs_dir)

@mcp.tool()
def calculate_bader_charge_from_cube(
    fcube: List[Path]|Path
) -> Dict[str, Any]:
    """
    Postprocess charge densities from ABACUS calculation.
    
    Parameters:
    fcube (str or list of str): Path to the cube file(s) containing the charge density.
    
    Returns:
    dict: A dictionary containing:
        - net_bader_charges: List of net Bader charge for each atom. Core charge is included.
        - bader_charges: List of Bader charge for each atom. The value represents the number of valence electrons for each atom, and core charge is not included.
        - atom_core_charges: List of core charge for each atom.
    """
    return _calculate_bader_charge_from_cube(fcube)
