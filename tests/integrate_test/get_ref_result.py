"""
An example result generation script
"""

from pathlib import Path
from abacusagent.env import set_envs, create_workpath
from abacusagent.modules.md import abacus_run_md
import os, shutil

set_envs()
create_workpath() # Allow submit to Bohrium by abacustest
print(os.getcwd())

test_path = Path(__file__).parent / 'abacus_inputs_dirs/H2/'
old_stru = test_path / "STRU_relaxed"
new_stru = test_path / "STRU"
shutil.copy(old_stru, new_stru)

force_thr_ev, stress_thr = 0.05, 1.0
outputs = abacus_run_md(test_path, md_type = 'nve', md_nstep = 10,
                        md_dt = 1.0, md_tfirst = 300)

os.unlink(new_stru)
print(outputs)
