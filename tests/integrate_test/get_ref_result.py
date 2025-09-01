"""
An example result generation script
"""

from pathlib import Path
from abacusagent.env import set_envs, create_workpath
from abacusagent.modules.band import abacus_cal_band
import os, shutil

set_envs()
create_workpath() # Allow submit to Bohrium by abacustest
print(os.getcwd())

test_path = Path(__file__).parent / 'abacus_inputs_dirs/Fe-BCC-prim/'
old_stru = test_path / "STRU_band"
new_stru = test_path / "STRU"
shutil.copy(old_stru, new_stru)

force_thr_ev, stress_thr = 0.05, 1.0
outputs = abacus_cal_band(test_path, mode='nscf')

os.unlink(new_stru)
print(outputs)
