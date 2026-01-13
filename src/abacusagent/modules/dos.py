from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.dos import abacus_dos_run as _abacus_dos_run

@mcp.tool()
def abacus_dos_run(
    abacus_inputs_dir: Path,
    pdos_mode: Literal['atoms', 'species', 'species+shell', 'species+orbital'] = 'species+shell',
    pdos_atom_indices: Optional[List[int]] = None,
    dos_edelta_ev: float = 0.01,
    dos_sigma: float = 0.07,
    dos_emin_ev: float = -10.0,
    dos_emax_ev: float = 10.0,
) -> Dict[str, Any]:
    """Run the DOS and PDOS calculation.
    
    This function will firstly run a SCF calculation with out_chg set to 1, 
    then run a NSCF calculation with init_chg set to 'file' and out_dos set to 1 or 2.
    If the INPUT parameter "basis_type" is "PW", then out_dos will be set to 1, and only DOS will be calculated and plotted.
    If the INPUT parameter "basis_type" is "LCAO", then out_dos will be set to 2, and both DOS and PDOS will be calculated and plotted.
    
    Args:
        abacus_inputs_dir: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        pdos_mode: Mode of plotted PDOS file.
            - "atoms": PDOS of a list of atoms will be plotted.
            - "species": Total PDOS of any species will be plotted in a picture.
            - "species+shell": PDOS for any shell (s, p, d, f, g,...) of any species will be plotted. PDOS of a shell of a species willbe plotted in a subplot.
            - "species+orbital": Orbital-resolved PDOS will be plotted. PDOS of orbitals in the same shell of a species will be plotted in a subplot.
        pdos_atom_indices: A list of atom indices, only used if pdos_mode is "atoms".
        dos_edelta_ev: Step size in writing Density of States (DOS) in eV.
        dos_sigma: Width of the Gaussian factor when obtaining smeared Density of States (DOS) in eV. 
        dos_emin_ev: Minimal range for Density of States (DOS) in eV. Default is -10.0.
        dos_emax_ev: Maximal range for Density of States (DOS) in eV. Default is 10.0.

    """
    return _abacus_dos_run(abacus_inputs_dir, pdos_mode, pdos_atom_indices, dos_edelta_ev, dos_sigma, dos_emin_ev, dos_emax_ev)
