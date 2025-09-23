s_block_elements = ['H', 'He']
p_block_elements = ["Li", "Be", "B", "C", "N", "O", "F", "Ne",
                    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                    "K", "Ca", "Ga", "Ge", "As", "Se", "Br", "Kr",
                    "Rb", "Sr", "In", "Sn", "Sb", "Te", "I", "Xe",
                    "Cs", "Ba", "Tl", "Pb", "Bi", "Po", "At", "Rn",
                    "Fr", "Ra", "Nh", "Fl", "Mc", "Lv", "Ts", "Og" ]
d_and_ds_block_elements = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                           "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                           "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                           "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"]
f_block_elements = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]
max_angular_momentum_of_elements = {}
for element in s_block_elements:
    max_angular_momentum_of_elements[element] = 's'
for element in p_block_elements:
    max_angular_momentum_of_elements[element] = 'p'
for element in d_and_ds_block_elements:
    max_angular_momentum_of_elements[element] = 'd'
for element in f_block_elements:
    max_angular_momentum_of_elements[element] = 'f'
