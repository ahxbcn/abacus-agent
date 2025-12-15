import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
import pytest
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.eos import abacus_eos

initilize_test_env()

@pytest.mark.long
class TestAbacusEos(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_nacl_prim = Path(__file__).parent / 'abacus_inputs_dirs/NaCl-prim/'
        self.stru_nacl_eos_cubic = self.abacus_inputs_dir_nacl_prim / "STRU_cubic_eos"
        self.input_nacl_eos = self.abacus_inputs_dir_nacl_prim / "INPUT_nspin1"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_abacus_eos_nacl_cubic(self):
        """
        Test the abacus_eos function for conventional cell of NaCl (the shape is cubic).
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl_prim, test_work_dir)
        shutil.copy2(self.stru_nacl_eos_cubic, test_work_dir / "STRU")
        shutil.copy2(self.input_nacl_eos, test_work_dir / "INPUT")

        outputs = abacus_eos(test_work_dir,
                             stru_scale_number = 5,
                             scale_stepsize = 0.02)

        print(outputs)

        self.assertIsInstance(outputs['eos_work_path'], get_path_type())
        self.assertIsInstance(outputs['eos_fig_path'], get_path_type())

        self.assertAlmostEqual(outputs['E0'], ref_results['E0'], places=3)
        self.assertAlmostEqual(outputs['V0'], ref_results['V0'], places=1)
        self.assertAlmostEqual(outputs['B0'], ref_results['B0'], places=1)
        self.assertAlmostEqual(outputs['B0_deriv'], ref_results['B0_deriv'], places=1)
