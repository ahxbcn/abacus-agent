import os
import shutil
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Union
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput, WriteKpt

from abacusagent.modules.util.comm import run_abacus, run_pyatb, collect_metrics
from abacusagent.modules.util.pyatb import property_calculation_scf

def get_high_symm_points_from_stru(stru_file: Path,
                                   stru_type: Literal['cif', 'poscar', 'abacus/stru'] = 'cif'
) -> Any:
    """
    Get high symmetry points and kpath from structure file.
    Args:
        stru_file (Path): Absolute path to the structure file.
        stru_type (Literal): Type of the structure file, can be 'cif', 'poscar', or 'abacus/stru'.
    Returns:
        A tuple containing high symmetry points and kpath.
    """
    import seekpath
    from ase.io import read

    if stru_type in ['cif', 'poscar']:
        stru = read(stru_file)
        cell = stru.get_cell()
        coord = stru.get_positions()
        labels = stru.get_chemical_symbols()
        label = []
        for i in labels:
            if i not in label:
                label.append(i)
        number = [label.index(i) for i in labels]
        stru_final = (cell, coord, number)
    elif stru_type == 'abacus/stru':
        stru = AbacusStru.ReadStru(stru_file)
        cell = stru.get_cell(bohr=True)
        coord = stru.get_coord(bohr=True)
        labels = stru.get_label()
        number = stru.get_label()
        label = []
        for i in labels: 
            if i not in label:
                label.append(i)
        number = [label.index(i) for i in labels]
        stru_final = (cell,coord,number)
    else:
        raise ValueError("stru_type should be 'cif', 'poscar', or 'abacus/stru'")
    
    sym_prec = 1e-5
    while sym_prec <= 1e-2:
        try:
            kpath = seekpath.get_path_orig_cell(stru_final,
                                                with_time_reversal=True,
                                                recipe='hpkot',
                                                threshold=1e-5,
                                                symprec=sym_prec,
                                                angle_tolerance=-1.0)
            break
        except:
            print("WARNING: get_path failed, increase symprec to %e" % (sym_prec*10))
            sym_prec *= 10
        
    if sym_prec > 1e-2:
        print(f"Symmetry precision {sym_prec} is too large")
    
    concatenated_kpath, path = [], None
    for path_idx, path_line in enumerate(kpath['path']):
        path_line_start, path_line_end = path_line[0], path_line[1]
        if path is None:
            path = [path_line_start, path_line_end]
        else:
            if path_idx == len(kpath['path']) - 1:
                if path is None:
                    path.append(path_line_start)
                path.append(path_line_end)
                concatenated_kpath.append(path)
                path = None
            elif path_idx < len(kpath['path']) - 1:
                if path is None:
                    path.append(path_line_start)
                path.append(path_line_end)
                next_line_start = kpath['path'][path_idx+1][0]
                if path_line_end != next_line_start:
                    concatenated_kpath.append(path)
                    path = None
            else:
                if path is None:
                    path = [path_line_start, path_line_end]
                else:
                    path.append(path_line_end)
                concatenated_kpath.append(path)
    
    kpath['path'] = concatenated_kpath
    return kpath
            
def get_high_symm_points_from_abacus_inputs_dir(abacus_inputs_dir: Path) -> Dict[str, Any]:
    """
    Get high symmetry points and kpath for STRU file in ABACUS inputs directory.
    """
    input_params = ReadInput(os.path.join(Path(abacus_inputs_dir), "INPUT"))
    stru_file = os.path.join(abacus_inputs_dir, input_params.get('stru_file', "STRU"))
    return get_high_symm_points_from_stru(stru_file, stru_type='abacus/stru')

def read_band_data(band_file: Path, efermi: float):
    """
    Read data in band file.
    Args:
        band_file (Path): Absolute path to the band file.
    Returns:
        A dictionary containing band data.
    Raises:
        RuntimeError: If read band data from BANDS_1.dat or BANDS_2.dat failed
    """
    bands, kline = [], []
    try:
        with open(band_file) as fin:
            for lines in fin:
                words = lines.split()
                nbands = len(words) - 2
                kline.append(float(words[1]))
                if len(bands) == 0:
                    for _ in range(nbands):
                        bands.append([])
            
                for i in range(nbands):
                    bands[i].append(float(words[i+2]) - efermi)
    except Exception as e:
        raise RuntimeError(f"Read data from {band_file} failed")
    
    return bands, kline, nbands
    
def split_array(array: List[Any], splits: List[int]):
    """
    Split band and kline by incontinuous points
    """
    splited_array = []
    start = 0

    for split_point in splits:
        splited_array.append(array[start:split_point])
        start = split_point
    
    splited_array.append(array[start:])
    return splited_array

def read_high_symmetry_labels(abacusjob_dir: Path):
    """
    Read high symmetry labels from KPT file
    """
    high_symm_labels = []
    band_point_nums = []
    band_point_num = 0
    with open(os.path.join(abacusjob_dir, "KPT")) as fin:
        for lines in fin:
            words = lines.split()
            if len(words) > 2:
                if words[-2] == '#':  # "# G" form
                    if words[-1] == 'G':
                        high_symm_labels.append(r'$\Gamma$')
                    else:
                        high_symm_labels.append(words[-1])
                    band_point_nums.append(band_point_num)
                    band_point_num += int(words[-3])
                elif words[-1].startswith("#"):  # "#G" form
                    label = words[-1][1:].split()[0]
                    if words[-1] == 'G':
                        high_symm_labels.append(r'$\Gamma$')
                    else:
                        high_symm_labels.append(label)
                    band_point_nums.append(band_point_num)
                    band_point_num += int(words[-2])
    
    return high_symm_labels, band_point_nums

def process_band_data(abacusjob_dir: Path, 
                      nspin: Literal[1, 2], 
                      efermi: float, 
                      kline: List[float],
                      bands: List[List[float]],  
                      bands_dw: Optional[List[List[float]]] = None):
    """
    Process band data, including properly process incontinous points and label high symmetry points
    """
    high_symm_labels, band_point_nums = read_high_symmetry_labels(abacusjob_dir)
    
    # Reduce extra kline length between incontinuous points
    modify_indexes = []
    for i in range(len(band_point_nums) - 1):
        if band_point_nums[i+1] - band_point_nums[i] == 1:
            reduce_length = kline[band_point_nums[i+1]] - kline[band_point_nums[i]]
            for j in range(band_point_nums[i+1], len(kline)):
                kline[j] -= reduce_length

            modify_indexes.append(i)
    
    # Modify incontinuous point labels
    high_symm_labels_old = high_symm_labels.copy()
    band_point_nums_old = band_point_nums.copy()
    high_symm_labels = []
    band_point_nums = []
    for i in range(len(high_symm_labels_old)):
        if i in modify_indexes:
            modified_tick = high_symm_labels_old[i] + "|" + high_symm_labels_old[i+1]
            high_symm_labels.append(modified_tick)
            band_point_nums.append(band_point_nums_old[i])
        elif i-1 in modify_indexes:
            pass
        else:
            band_point_nums.append(band_point_nums_old[i])
            high_symm_labels.append(high_symm_labels_old[i])
    
    # Split incontinuous bands to list of continous bands
    band_split_points = [band_point_nums_old[x]+1 for x in modify_indexes]
    kline_splited = split_array(kline, band_split_points)
    bands_splited = []
    for i in range(len(bands)):
        bands_splited.append(split_array(bands[i], band_split_points))
    if nspin == 2:
        bands_dw_splited = []
        for i in range(len(bands_dw)):
            bands_dw_splited.append(split_array(bands_dw[i], band_split_points))

    high_symm_poses = [kline[i] for i in band_point_nums]
    
    if nspin == 1:
        return high_symm_labels, high_symm_poses, kline_splited, bands_splited
    else:
        return high_symm_labels, high_symm_poses, kline_splited, bands_splited, bands_dw_splited

def abacus_plot_band_nscf(abacusjob_dir: Path,
                          energy_min: float = -10,
                          energy_max: float = 10
) -> Dict[str, Any]:
    """
    Plot band after ABACUS SCF and NSCF calculation.
    Args:
        abacusjob_dir (str): Absolute path to the ABACUS calculation directory.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
    Returns:
        A dictionary containing band gap of the system and path to the plotted band.
    Raises:
        NotImplementedError: If band plot for an nspin=4 calculation is requested
        RuntimeError: If read band data from BANDS_1.dat or BANDS_2.dat failed
    """
    import matplotlib.pyplot as plt

    input_args = ReadInput(os.path.join(abacusjob_dir, "INPUT"))
    suffix = input_args.get('suffix', 'ABACUS')
    nspin = input_args.get('nspin', 1)
    if nspin not in (1, 2):
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")
    
    metrics = collect_metrics(abacusjob_dir, ['efermi', 'nelec', 'band_gap'])
    efermi, band_gap = metrics['efermi'], float(metrics['band_gap'])
    band_file = os.path.join(abacusjob_dir, f"OUT.{suffix}/BANDS_1.dat")
    if nspin == 2:
        band_file_dw = os.path.join(abacusjob_dir, f"OUT.{suffix}/BANDS_2.dat")
    
    # Read band data
    bands, kline, nbands = read_band_data(band_file, efermi)
    if nspin == 2:
        bands_dw, _, _ = read_band_data(band_file_dw, efermi)
    
    # Process band data
    if nspin == 1:
        high_symm_labels, high_symm_poses, kline_splited, bands_splited = \
            process_band_data(abacusjob_dir, nspin, efermi, kline, bands)
    else:
        high_symm_labels, high_symm_poses, kline_splited, bands_splited, bands_dw_splited = \
            process_band_data(abacusjob_dir, nspin, efermi, kline, bands, bands_dw)
    
    # Final band plot
    for i in range(nbands):
        for j in range(len(kline_splited)):
            plt.plot(kline_splited[j], bands_splited[i][j], 'r-', linewidth=1.0)
    if nspin == 2:
        for i in range(nbands):
            for j in range(len(kline_splited)):
                plt.plot(kline_splited[j], bands_dw_splited[i][j], 'b--', linewidth=1.0)
    plt.xlim(kline[0], kline[-1])
    plt.ylim(energy_min, energy_max)
    plt.ylabel(r"$E-E_\text{F}$/eV")
    plt.xticks(high_symm_poses, high_symm_labels)
    plt.grid()
    plt.title(f"Band structure  (Gap = {band_gap:.2f} eV)")
    plt.savefig(os.path.join(abacusjob_dir, 'band.png'), dpi=300)
    plt.close()

    return {'band_gap': band_gap,
            'band_picture': Path(os.path.join(abacusjob_dir, 'band.png')).absolute()}

def write_pyatb_input(band_calc_path: Path):
    """
    Write Input file for PYATB
    """
    input_args = ReadInput(os.path.join(band_calc_path, "INPUT"))
    suffix = input_args.get('suffix', 'ABACUS')
    nspin = input_args.get('nspin', 1)
    metrics = collect_metrics(band_calc_path, ['efermi', 'cell', 'band_gap'])
    efermi, cell = metrics['efermi'], metrics['cell']

    input_parameters = {
        'nspin': nspin,
        'package': "ABACUS",
        'fermi_energy': efermi,
        'HR_route': f"OUT.{suffix}/data-HR-sparse_SPIN0.csr",
        'SR_route': f"OUT.{suffix}/data-SR-sparse_SPIN0.csr",
        'rR_route': f"OUT.{suffix}/data-rR-sparse.csr",
        "HR_unit":  "Ry",
        "rR_unit": "Bohr"
    }
    if nspin == 2:
        input_parameters['HR_route'] += f' OUT.{suffix}/data-HR-sparse_SPIN1.csr'
        input_parameters['SR_route'] += f' OUT.{suffix}/data-SR-sparse_SPIN1.csr'
    
    shutil.move(os.path.join(band_calc_path, "INPUT"), os.path.join(band_calc_path, "INPUT_scf"))
    shutil.move(os.path.join(band_calc_path, "KPT"),   os.path.join(band_calc_path, "KPT_scf"))
    pyatb_input_file = open(os.path.join(band_calc_path, "Input"), "w")
    
    pyatb_input_file.write("INPUT_PARAMETERS\n{\n")
    for key, value in input_parameters.items():
        pyatb_input_file.write(f"    {key}  {value}\n")
    pyatb_input_file.write("}\n\nLATTICE\n{\n")

    pyatb_input_file.write(f"    {'lattice_constant'}  {1.8897162}\n")
    pyatb_input_file.write(f"    {'lattice_constant_unit'}  {'Bohr'}\n    lattice_vector\n")
    for cell_vec in cell:
        pyatb_input_file.write(f"    {cell_vec[0]:.8f}  {cell_vec[1]:.8f}  {cell_vec[2]:.8f}\n")
    pyatb_input_file.write("}\n\nBAND_STRUCTURE\n{\n    kpoint_mode   line\n")

    # Get kline and write to pyatb Input file
    kpt_file = os.path.join(band_calc_path, "KPT_band")
    kpt_file_content = []
    with open(kpt_file) as fin:
        for lines in fin:
            words = lines.split()
            kpt_file_content.append(words)

    high_symm_nums = int(kpt_file_content[1][0])
    kpoint_label = ''
    for linenum in range(3, 3+high_symm_nums):
        kpoint_label += kpt_file_content[linenum][-1].split('#')[-1]
        if linenum < 2+high_symm_nums:
            kpoint_label += ", "
    pyatb_input_file.write(f"    kpoint_num    {high_symm_nums}\n")
    pyatb_input_file.write(f"    kpoint_label  {kpoint_label}\n    high_symmetry_kpoint\n")
    for linenum in range(3, 3+high_symm_nums):
        kpoint_coord = f"    {kpt_file_content[linenum][0]} {kpt_file_content[linenum][1]} {kpt_file_content[linenum][2]}"
        kline_num = f" {kpt_file_content[linenum][3]}\n"
        pyatb_input_file.write(kpoint_coord + kline_num)
    pyatb_input_file.write("}\n")

    pyatb_input_file.close()

    return True

def abacus_plot_band_pyatb(band_calc_path: Path,
                           energy_min: float = -10,
                           energy_max: float = 10,
) -> Dict[str, Any]:
    """
    Read result from self-consistent (scf) calculation of hybrid functional using uniform grid,
    and calculate and plot band using PYATB.  

    Currently supports only non-spin-polarized and collinear spin-polarized calculations.

    Args:
        band_calc_path (str): Absolute path to the band calculation directory.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.

    Returns:
        dict: A dictionary containing:
            - 'band_gap': Calculated band gap in eV. 
            - 'band_picture': Path to the saved band structure plot image file.
    Raises:
        NotImplementedError: If requestes to plot band structure for a collinear or SOC calculation
        RuntimeError: If read band gap from band_info.dat failed
    """
    input_args = ReadInput(os.path.join(band_calc_path, "INPUT"))
    nspin = input_args.get('nspin', 1)
    band_gap = float(collect_metrics(band_calc_path, ['band_gap'])['band_gap'])
    if nspin not in (1, 2):
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")
    
    if write_pyatb_input(band_calc_path) is not True:
        raise RuntimeError("Failed to write pyatb input file")
    
    # Use pyatb to plot band
    run_pyatb(band_calc_path)

    # read band gap
    band_gaps = []
    try:
        with open(os.path.join(band_calc_path, "Out/Band_Structure/band_info.dat")) as fin:
            for lines in fin:
                if "Band gap" in lines:
                    band_gaps.append(float(lines.split()[-1]))
    except Exception as e:
        raise RuntimeError("band_info.dat not found!")
    
    # Modify auto generated plot_band.py and replot the band
    os.system(f'sed -i "16c y_min =  {energy_min} # eV" {band_calc_path}/Out/Band_Structure/plot_band.py')
    os.system(f'sed -i "17c y_max =  {energy_max} # eV" {band_calc_path}/Out/Band_Structure/plot_band.py')
    os.system(f'''sed -i "18c fig_name = os.path.join(work_path, \\"band.png\\")" "{band_calc_path}/Out/Band_Structure/plot_band.py"''')
    os.system(f'sed -i "91c plt.savefig(fig_name, dpi=300)" {band_calc_path}/Out/Band_Structure/plot_band.py')
    os.system(f"cd {band_calc_path}/Out/Band_Structure; python plot_band.py; cd ../../")
    
    # Copy plotted band.pdf to given directory
    band_picture = os.path.join(band_calc_path, "band.png")
    os.system(f"cp {os.path.join(band_calc_path, 'Out/Band_Structure/band.png')} {band_picture}")

    return {'band_gap': band_gap,
            'band_picture': Path(band_picture).absolute()}    

def abacus_cal_band(abacus_inputs_dir: Path,
                    mode: Literal["nscf", "pyatb", "auto"] = "auto",
                    kpath: Union[List[str], List[List[str]]] = None,
                    high_symm_points: Dict[str, List[float]] = None,
                    energy_min: float = -10,
                    energy_max: float = 10,
                    insert_point_nums: int = 30
) -> Dict[str, float|str]:
    """
    Calculate band using ABACUS based on prepared directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    PYATB or ABACUS NSCF calculation will be used according to parameters in INPUT.
    Args:
        abacus_inputs_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
        mode: Method used to plot band. Should be `auto`, `pyatb` or `nscf`. 
            - `nscf` means using `nscf` calculation in ABACUS to calculate and plot the band
            - `pyatb` means using PYATB to plot the band
            - `auto` means deciding use `nscf` or `pyatb` mode according to the `basis_type` in INPUT file and files included in `abacus_inputs_dir`.
                -- If charge files are in `abacus_input_dir`, `nscf` mode will be used.
                -- If matrix files are in `abacus_input_dir`, `pyatb` mode will be used.
                -- If no matrix file or charge file are in `abacus_input_dir`, will determine mode by `basis_type`. If `basis_type` is lcao, will use `pyatb` mode.
                    If `basis_type` is pw, will use `nscf` mode.
        kpath (Tuple[List[str], List[List[str]]]): 
                A list of name of high symmetry points in the band path. Non-continuous line of high symmetry points are stored as seperate lists.
                For example, ['G', 'M', 'K', 'G'] and [['G', 'X', 'P', 'N', 'M', 'S'], ['S_0', 'G', R']] are both acceptable inputs.
                Default is None. If None, will use automatically generated kpath.
                `kpath` must be used with `high_symm_points` to take effect.
        high_symm_points: A dictionary containing high symmetry points and their coordinates in the band path. All points in `kpath` should be included.
                For example, {'G': [0, 0, 0], 'M': [0.5, 0.0, 0.0], 'K': [0.33333333, 0.33333333, 0.0], 'G': [0, 0, 0]}.
                Default is None. If None, will use automatically generated high symmetry points.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
        insert_point_nums (int): Number of points to insert between two high symmetry points. Default is 30.
    Returns:
        A dictionary containing band gap, path to the work directory for calculating band and path to the plotted band.
    Raises:
    """
    try:
        input_params = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
        original_stru_file = os.path.join(abacus_inputs_dir, input_params.get('stru_file', "STRU"))
        original_stru = AbacusStru.ReadStru(original_stru_file)
        band_kpt_file = os.path.join(abacus_inputs_dir, "KPT_band")
        new_stru, point_coords, path, _ = original_stru.get_kline(point_number=30,
                                                                      new_stru_file=original_stru_file,
                                                                      kpt_file=band_kpt_file)
        
        if kpath is not None and high_symm_points is not None:
            kline = []
            if all(isinstance(item, str) for item in kpath): # A whole continous kline
                for idx, high_symm_point in enumerate(kpath):
                    if idx == len(kpath) - 1: # Treat tail of kline
                        kpoint = high_symm_points[high_symm_point] + [1, '# ' + high_symm_point]
                    else:
                        kpoint = high_symm_points[high_symm_point] + [insert_point_nums, '# ' + high_symm_point]
                    kline.append(kpoint)
            elif all(isinstance(item, list) for item in kpath): # kline with uncontinous points
                kline = []
                for sub_kpath in kpath:
                    for idx, high_symm_point in enumerate(sub_kpath):
                        if idx == len(sub_kpath) - 1: # Treat tail of kline
                            kpoint = high_symm_points[high_symm_point] + [1, '# ' + high_symm_point]
                        else:
                            kpoint = high_symm_points[high_symm_point] + [insert_point_nums, '# ' + high_symm_point]
                        kline.append(kpoint)
            
            WriteKpt(kline, band_kpt_file, model='line')
        elif kpath is not None or high_symm_points is not None:
            print("kpath and high_symm_points must be used together. Use auto-generated kpath and high_symm_points")
        
        force_run = True if original_stru.get_natoms() != new_stru.get_natoms() else False
        scf_output = property_calculation_scf(abacus_inputs_dir, mode, always_run=force_run)
        work_path, mode = scf_output["work_path"], scf_output["mode"]
        if mode == 'pyatb':
            # Obtain band using PYATB
            postprocess_output = abacus_plot_band_pyatb(work_path,
                                                        energy_min,
                                                        energy_max)

            return {'band_gap': postprocess_output['band_gap'],
                    'band_calc_dir': abacus_inputs_dir,
                    'band_picture': postprocess_output['band_picture'],
                    "message": "The band is calculated using PYATB after SCF calculation using ABACUS"}

        elif mode == 'nscf':
            input_params["calculation"] = "nscf"
            input_params["init_chg"] = "file"
            input_params["out_band"] = 1
            input_params["symmetry"] = 0
            input_params['kspacing'] = None
            WriteInput(input_params, os.path.join(work_path, "INPUT"))
            
            # Prepare line-mode KPT file
            kpt_file = os.path.join(work_path, input_params.get('kpt_file', 'KPT'))
            shutil.copy(band_kpt_file, kpt_file)

            run_abacus(work_path)

            plot_output = abacus_plot_band_nscf(work_path, energy_min, energy_max)

            return {'band_gap': plot_output['band_gap'],
                    'band_calc_dir': Path(work_path).absolute(),
                    'band_picture': Path(plot_output['band_picture']).absolute(),
                    "message": "The band structure is computed via a non-self-consistent field (NSCF) calculation using ABACUS, following a converged self-consistent field (SCF) calculation."}
        else:
            raise ValueError(f"Calculation mode {mode} not in ('pyatb', 'nscf', 'auto')")
    except Exception as e:
        return {'message': f"Calculating band failed: {e}"}
