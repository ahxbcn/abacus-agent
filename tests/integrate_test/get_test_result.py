"""
An example result generation script
"""

from pathlib import Path
from abacusagent.env import set_envs
from abacusagent.modules.relax import abacus_do_relax
import shutil

set_envs()

test_path = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim/'
old_stru = test_path / "STRU_no_relax_cell"
new_stru = test_path / "STRU"
shutil.copy(old_stru, new_stru)

force_thr_ev, stress_thr = 0.05, 1.0
outputs = abacus_do_relax(test_path,
                          force_thr_ev = force_thr_ev,
                          max_steps = 10,
                          relax_cell = False,
                          relax_method = 'cg',
                          relax_new = False)

print(outputs)
