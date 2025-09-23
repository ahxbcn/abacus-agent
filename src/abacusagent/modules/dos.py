from pathlib import Path
from typing import Dict, Any, List, Literal

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.dos import abacus_dos_run as _abacus_dos_run
@mcp.tool()
def abacus_dos_run(
    abacus_inputs_dir: Path,
    pdos_mode: Literal['species', 'species+shell', 'species+orbital'] = 'species+shell',
    dos_edelta_ev: float = 0.01,
    dos_sigma: float = 0.07,
    dos_scale: float = 0.01,
    dos_emin_ev: float = None,
    dos_emax_ev: float = None,
    dos_nche: int = None,
) -> Dict[str, Any]:
    """Run the DOS and PDOS calculation.
    
    This function will firstly run a SCF calculation with out_chg set to 1, 
    then run a NSCF calculation with init_chg set to 'file' and out_dos set to 1 or 2.
    If the INPUT parameter "basis_type" is "PW", then out_dos will be set to 1, and only DOS will be calculated and plotted.
    If the INPUT parameter "basis_type" is "LCAO", then out_dos will be set to 2, and both DOS and PDOS will be calculated and plotted.
    
    Args:
        abacus_inputs_dir: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        pdos_mode: Mode of plotted PDOS file.
            - "species": Total PDOS of any species will be plotted in a picture.
            - "species+shell": PDOS for any shell (s, p, d, f, g,...) of any species will be plotted. PDOS of a shell of a species willbe plotted in a subplot.
            - "species+orbital": Orbital-resolved PDOS will be plotted. PDOS of orbitals in the same shell of a species will be plotted in a subplot.
        dos_edelta_ev: Step size in writing Density of States (DOS) in eV.
        dos_sigma: Width of the Gaussian factor when obtaining smeared Density of States (DOS) in eV. 
        dos_scale: Defines the energy range of DOS output as (emax-emin)*(1+dos_scale), centered at (emax+emin)/2. 
                   This parameter will be used when dos_emin_ev and dos_emax_ev are not set.
        dos_emin_ev: Minimal range for Density of States (DOS) in eV.
        dos_emax_ev: Maximal range for Density of States (DOS) in eV.
        dos_nche: The order of Chebyshev expansions when using Stochastic Density Functional Theory (SDFT) to calculate DOS.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - dos_fig_path: Path to the plotted DOS.
            - pdos_fig_path: Path to the plotted PDOS. Only for LCAO basis.
            - scf_work_path: Path to the work directory of SCF calculation.
            - scf_normal_end: If the SCF calculation ended normally.
            - scf_steps: Number of steps of SCF iteration.
            - scf_converge: If the SCF calculation converged.
            - scf_energy: The calculated energy of SCF calculation.
            - nscf_work_path: Path to the work directory of NSCF calculation.
            - nscf_normal_end: If the SCF calculation ended normally.
    """
    return _abacus_dos_run(abacus_inputs_dir, pdos_mode, dos_edelta_ev, dos_sigma, dos_scale, dos_emin_ev, dos_emax_ev, dos_nche)
