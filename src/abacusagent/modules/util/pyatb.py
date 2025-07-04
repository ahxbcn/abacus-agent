"""
Use Pyatb to do property calculation.
"""
import os
from pathlib import Path
from typing import Dict, Any, Literal, List
from abacusagent.modules.util.comm import (
    generate_work_path, 
    link_abacusjob, 
    run_abacus, 
    run_command,
    has_chgfile, 
    has_pyatb_matrix_files,
    get_physical_cores
)
from abacusagent.init_mcp import mcp

from abacustest.lib_prepare.abacus import ReadInput, WriteInput, AbacusStru
from abacustest.lib_collectdata.collectdata import RESULT


def property_calculation_scf(
    abacus_inputs_path: Path,
    mode: Literal["nscf", "pyatb", "auto"] = "auto",
                    ):
    """Perform the SCF calculation for property calculations like DOS or band structure.

    Args:
        abacus_inputs_path (Path): Path to the ABACUS input files.
        mode (Literal["nscf", "pyatb", "auto"]): Mode of operation, default is "auto".
            nscf: first run SCF with out_chg=1, then run nscf with init_chg=file.
            pyatb: run SCF with out_mat_r and out_mat_hs2 = 1, then calculate properties using Pyatb.
            auto: automatically determine the mode based on the input parameters. If basis is LCAO, use "pyatb", otherwise use "nscf".

    Returns:
        Dict[str, Any]: A dictionary containing the work path, normal end status, SCF steps, convergence status, and energies.
    """

    input_param = ReadInput(os.path.join(abacus_inputs_path, 'INPUT'))
    basis_type = input_param.get("basis_type", "pw")
    if mode == "auto":
        if basis_type.lower() == "lcao":
            mode = "pyatb"
        else:
            mode = "nscf"
    
    if basis_type == "pw" and mode == "pyatb":
        raise ValueError("Pyatb mode is not supported for PW basis. Please use 'nscf' mode instead.")

    if (mode == "nscf" and has_chgfile(abacus_inputs_path)) or (mode == "pyatb" and has_pyatb_matrix_files(abacus_inputs_path)):
        print("Charge or matrix files already exist, skipping SCF calculation.")
        work_path = abacus_inputs_path
    else:
        work_path = generate_work_path()
        link_abacusjob(src=abacus_inputs_path,
                       dst=work_path,
                       copy_files=["INPUT"])
        if mode == "nscf":
            input_param["calculation"] = "scf"
            input_param["out_chg"] = 1
        elif mode == "pyatb":
            input_param["calculation"] = "scf"
            input_param["out_mat_r"] = 1
            input_param["out_mat_hs2"] = 1
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'nscf', 'pyatb', or 'auto'.")
        
        WriteInput(input_param, os.path.join(work_path, 'INPUT'))
        run_abacus(work_path, input_param.get("suffix", "ABACUS"))
        
    rs = RESULT(path=work_path, fmt="abacus")

    return {
        "work_path": Path(work_path).absolute(),
        "normal_end": rs["normal_end"],
        "scf_steps": rs["scf_steps"],
        "converge": rs["converge"],
        "energies": rs["energies"],
        "mode": mode
    }
        
#@mcp.tool()
def plot_fatband_pyatb(
    abacus_inputs_path: Path,
    fatband_element_orbital: Dict[str, Literal['s', 'p', 'd']] = None,
    energy_range: float = 5.0
) -> Dict[str, Any]:
    """
    Use pyatb to plot fat band from previous ABACUS SCF calculation.
    Args:
        abacus_inputs_path (Path): Path to the directory of output files for ABACUS calculation.
        fatband_element_orbital (Dict[str, Literal['s', 'p', 'd']]): Element and orbital for plotting fat band.
        energy_range (float): Energy range for plotting fat band.
    Returns:
        Dict[str, Any]: A dictionary containing the work path of pyatb and plotted fatband.
    """
    abacus_inputs_path = Path(abacus_inputs_path).absolute()
    print(os.getcwd())
    os.chdir(abacus_inputs_path)
    print(os.getcwd())
    a = input()
    pyatb_command = f"pyatb_input -i {abacus_inputs_path} --erange {energy_range} --fatband"
    print(pyatb_command)
    return_code, out, err = run_command(pyatb_command)
    if return_code != 0:
        raise RuntimeError(f"Preparing pyatb Input file failed with return code {return_code}, out: {out}, err: {err}")
    
    physical_cores = get_physical_cores()
    pyatb_command = f"cd pyatb; export OMP_NUM_THREADS=1; mpirun -np {physical_cores} pyatb"
    return_code, out, err = run_command(pyatb_command)
    
    if return_code != 0:
        raise RuntimeError(f"pyatb failed with return code {return_code}, out: {out}, err: {err}")
    
    input_params = ReadInput(os.path.join(abacus_inputs_path, "INPUT"))
    nspin = input_params.get("nspin", 1)

    return_dict = {"pyatb_cal_path": Path(os.path.join(abacus_inputs_path, "pyatb")).absolute()}

    fatband_definition = ''
    orbital_labels = {'s': 0, 'p': 1, 'd': 2}
    for key, value in fatband_element_orbital.items():
        fatband_definition += f"\"{key}\": [{orbital_labels[value]}],"
    if nspin == 1:
        run_command(f"sed -i '35c species = {{{fatband_definition}}}' pyatb/Out/Fat_Band/plot_fatband.py")
        run_command(f"sed -i '43c pband.write(species=species)' pyatb/Out/Fat_Band/plot_fatband.py")
        run_command(f"sed -i '48c pband.plot_contributions(fig, ax, species=species, efermi=efermi, energy_range=energy_range, colors=[])' pyatb/Out/Fat_Band/plot_fatband.py")
        run_command("cd pyatb/Out/Fat_Band; python plot_fatband.py")
        os.chdir(abacus_inputs_path)

        return_dict['fatband_picture'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "Fat_Band", "fatband.png")).absolute()
    elif nspin == 2:
        # Plot fatband for up spin
        run_command(f"sed -i \"s/fatband.xml'/fatband_up.xml'/g\" pyatb/Out/Fat_Band/plot_fatband.py")
        run_command(f"sed -i '35c species = {{{fatband_definition}}}' pyatb/Out/Fat_Band/plot_fatband.py")
        run_command(f"sed -i '43c pband.write(species=species)' pyatb/Out/Fat_Band/plot_fatband.py")
        run_command(f"sed -i '48c pband.plot_contributions(fig, ax, species=species, efermi=efermi, energy_range=energy_range, colors=[])' pyatb/Out/Fat_Band/plot_fatband.py")
        run_command(f"sed -i \"53c fig.savefig('fatband_up.png')\" pyatb/Out/Fat_Band/plot_fatband.py")
        run_command("cd pyatb/Out/Fat_Band; python plot_fatband.py")
        
        # Plot fatband for down spin
        run_command(f"sed -i \"s/fatband_up.xml'/fatband_dn.xml'/g\" pyatb/Out/Fat_Band/plot_fatband.py")
        run_command(f"sed -i \"53c fig.savefig('fatband_dn.png')\" pyatb/Out/Fat_Band/plot_fatband.py")
        run_command("cd pyatb/Out/Fat_Band; python plot_fatband.py")
        
        return_dict['fatband_picture_up'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "Fat_Band", "fatband_up.png")).absolute()
        return_dict['fatband_picture_down'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "Fat_Band", "fatband_dn.png")).absolute()
    else:
        raise ValueError("nspin 4 not supported yet")

    return return_dict

#@mcp.tool()
def plot_pdos_pyatb(
    abacus_inputs_path: Path,
    pdos_element_orbital: Dict[str, Literal['s', 'p', 'd']] = None,
    energy_range: float = 5.0
) -> Dict[str, Any]:
    """
    Use pyatb to plot PDOS from previous ABACUS SCF calculation.
    Args:
        abacus_inputs_path (Path): Path to the directory of output files for ABACUS calculation.
        pdos_element_orbital (Dict[str, Literal['s', 'p', 'd']]): Element and orbital for plotting pdos.
        energy_range (float): Energy range for plotting pdos.
    Returns:
        Dict[str, Any]: A dictionary containing the work path of pyatb and plotted tdos and pdos figures.
    """
    os.chdir(abacus_inputs_path)
    pyatb_command = f"pyatb_input --erange {energy_range} --pdos"
    return_code, out, err = run_command(pyatb_command)
    if return_code != 0:
        raise RuntimeError(f"Preparing pyatb Input file failed with return code {return_code}, out: {out}, err: {err}")
    
    physical_cores = get_physical_cores()
    pyatb_command = f"cd pyatb; export OMP_NUM_THREADS=1; mpirun -np {physical_cores} pyatb"
    return_code, out, err = run_command(pyatb_command)
    
    if return_code != 0:
        raise RuntimeError(f"pyatb failed with return code {return_code}, out: {out}, err: {err}")
    
    input_params = ReadInput(os.path.join(abacus_inputs_path, "INPUT"))
    nspin = input_params.get("nspin", 1)

    return_dict = {"pyatb_cal_path": Path(os.path.join(abacus_inputs_path, "pyatb")).absolute()}
    
    orbital_labels = {'s': 0, 'p': 1, 'd': 2}
    pdos_definition = ''
    for element, orbital in pdos_element_orbital.items():
        pdos_definition += f"\"{element}\": [{orbital_labels[orbital]}], "
    run_command(f"sed -i '''45c species = {{{pdos_definition}}}''' pyatb/Out/PDOS/plot_dos.py")
    if nspin == 2:
        run_command("sed -i '11c dos_range = [-5, 5]' pyatb/Out/PDOS/plot_dos.py")
        run_command("sed -i '49c dos_range = [-5, 5]' pyatb/Out/PDOS/plot_dos.py")
    run_command(f"sed -i '''55c pdos.write(species=species)''' pyatb/Out/PDOS/plot_dos.py")
    run_command("sed -i '58c dosplots = pdos.plot(fig, ax, species=species, efermi=efermi, shift=False, energy_range=energy_range, dos_range=dos_range)' pyatb/Out/PDOS/plot_dos.py")

    run_command("cd pyatb/Out/PDOS; python plot_dos.py")
    os.chdir(abacus_inputs_path)
    
    return_dict['tdos_picture'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "PDOS", "tdos.png")).absolute()
    return_dict['pdos_picture'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "PDOS", "pdos.png")).absolute()

    return return_dict

#@mcp.tool()
def property_calculation_pyatb(
    abacus_inputs_path: Path,
    band: bool = False,
    pdos: bool = False,
    pdos_element_orbital: Dict[str, Literal['s', 'p', 'd']] = None,
    fatband: bool = False,
    fatband_element_orbital: Dict[str, Literal['s', 'p', 'd']] = None,
    energy_range: float = 5.0
) -> Dict[str, Any]:
    """
    Use pyatb to calculate band from previous ABACUS SCF calculation.
    Args:
        abacus_inputs_path (Path): Path to the directory of output files for ABACUS calculation.
        band (bool): Whether plot the band.
        pdos (bool): Whether plot the pdos.
        pdos_element_orbital (Dict[str, Literal['s', 'p', 'd']]): Element for plotting PDOS.
        fat_band (bool): Whether plot the fat band.
        fatband_element_orbital (Dict[str, Literal['s', 'p', 'd']]): Element and orbital for plotting fat band.
        energy_range (float): Plot band, pdos, fat band in the energy range of (-energy_range, energy_range) respect to fermi energy.
    Returns:
        A dictionary containing output files of pyatb and the plotted figures.
    """
    os.chdir(abacus_inputs_path)
    pyatb_command = f"pyatb_input --erange {energy_range} "
    if band:
        pyatb_command += "--band "
    if pdos:
        pyatb_command += "--pdos "
    if fatband:
        pyatb_command += "--fatband "
    
    return_code, out, err = run_command(pyatb_command)
    if return_code != 0:
        raise RuntimeError(f"Preparing pyatb Input file failed with return code {return_code}, out: {out}, err: {err}")
    
    physical_cores = get_physical_cores()
    pyatb_command = f"cd pyatb; export OMP_NUM_THREADS=1; mpirun -np {physical_cores} pyatb"
    return_code, out, err = run_command(pyatb_command)
    
    if return_code != 0:
        raise RuntimeError(f"pyatb failed with return code {return_code}, out: {out}, err: {err}")
    
    input_params = ReadInput(os.path.join(abacus_inputs_path, "INPUT"))
    nspin = input_params.get("nspin", 1)

    return_dict = {"pyatb_cal_path": Path(os.path.join(abacus_inputs_path, "pyatb")).absolute()}
    if band:
        return_dict['band_picture'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "Band_Structure", "band.pdf")).absolute()   
    
    # Plot PDOS for specified element and orbital
    orbital_labels = {'s': 0, 'p': 1, 'd': 2}
    if pdos:
        pdos_definition = ''
        for element, orbital in pdos_element_orbital.items():
            pdos_definition += f"\"{element}\": [{orbital_labels[orbital]}], "
        run_command(f"sed -i '''45c species = {{{pdos_definition}}}''' pyatb/Out/PDOS/plot_dos.py")
        if nspin == 2:
            run_command("sed -i '11c dos_range = [-5, 5]' pyatb/Out/PDOS/plot_dos.py")
            run_command("sed -i '49c dos_range = [-5, 5]' pyatb/Out/PDOS/plot_dos.py")
        run_command(f"sed -i '''55c pdos.write(species=species)''' pyatb/Out/PDOS/plot_dos.py")
        run_command("sed -i '58c dosplots = pdos.plot(fig, ax, species=species, efermi=efermi, shift=False, energy_range=energy_range, dos_range=dos_range)' pyatb/Out/PDOS/plot_dos.py")

        run_command("cd pyatb/Out/PDOS; python plot_dos.py")
        os.chdir(abacus_inputs_path)
        
        return_dict['tdos_picture'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "PDOS", "tdos.png")).absolute()
        return_dict['pdos_picture'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "PDOS", "pdos.png")).absolute()
    
    # Plot fat band for specified element and orbital
    # Currently assuming only 1 element and 1 orbital are considered in fat band calculation
    if fatband:
        fatband_definition = ''
        for key, value in fatband_element_orbital.items():
            fatband_definition += f"\"{key}\": [{orbital_labels[value]}],"
        if nspin == 1:
            run_command(f"sed -i '35c species = {{{fatband_definition}}}' pyatb/Out/Fat_Band/plot_fatband.py")
            run_command(f"sed -i '43c pband.write(species=species)' pyatb/Out/Fat_Band/plot_fatband.py")
            run_command(f"sed -i '48c pband.plot_contributions(fig, ax, species=species, efermi=efermi, energy_range=energy_range, colors=[])' pyatb/Out/Fat_Band/plot_fatband.py")
            run_command("cd pyatb/Out/Fat_Band; python plot_fatband.py")
            os.chdir(abacus_inputs_path)

            return_dict['fatband_picture'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "Fat_Band", "fatband.png")).absolute()
        elif nspin == 2:
            # Plot fatband for up spin
            run_command(f"sed -i \"s/fatband.xml'/fatband_up.xml'/g\" pyatb/Out/Fat_Band/plot_fatband.py")
            run_command(f"sed -i '35c species = {{{fatband_definition}}}' pyatb/Out/Fat_Band/plot_fatband.py")
            run_command(f"sed -i '43c pband.write(species=species)' pyatb/Out/Fat_Band/plot_fatband.py")
            run_command(f"sed -i '48c pband.plot_contributions(fig, ax, species=species, efermi=efermi, energy_range=energy_range, colors=[])' pyatb/Out/Fat_Band/plot_fatband.py")
            run_command(f"sed -i \"53c fig.savefig('fatband_up.png')\" pyatb/Out/Fat_Band/plot_fatband.py")
            run_command("cd pyatb/Out/Fat_Band; python plot_fatband.py")
            
            # Plot fatband for down spin
            run_command(f"sed -i \"s/fatband_up.xml'/fatband_dn.xml'/g\" pyatb/Out/Fat_Band/plot_fatband.py")
            run_command(f"sed -i \"53c fig.savefig('fatband_dn.png')\" pyatb/Out/Fat_Band/plot_fatband.py")
            run_command("cd pyatb/Out/Fat_Band; python plot_fatband.py")
            
            return_dict['fatband_picture_up'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "Fat_Band", "fatband_up.png")).absolute()
            return_dict['fatband_picture_down'] = Path(os.path.join(abacus_inputs_path, "pyatb", "Out", "Fat_Band", "fatband_dn.png")).absolute()
        else:
            raise ValueError("nspin 4 not supported yet")

    return return_dict
