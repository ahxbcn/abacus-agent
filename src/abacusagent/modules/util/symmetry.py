import os

from pathlib import Path
from typing import Literal, Dict, Any
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput

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
