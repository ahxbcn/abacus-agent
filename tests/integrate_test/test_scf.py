import os
import shutil
from pathlib import Path
import unittest
import tempfile
import pytest
import inspect
from abacusagent.modules.scf import abacus_calculation_scf
from utils import initilize_test_env, load_test_ref_result

initilize_test_env()

class TestAbacusCalculationScf(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_si_prim = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim/'
        self.stru_scf = self.abacus_inputs_dir_si_prim / "STRU_scf"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)


    def tearDown(self):
        os.chdir(self.original_cwd)
    
    @pytest.mark.smoke
    def test_abacus_calculation_scf(self):
        """
        Test the abacus_calculation_scf function.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = abacus_calculation_scf(test_work_dir)
        scf_work_dir = Path(outputs['scf_work_dir']).absolute()

        self.assertTrue(os.path.exists(scf_work_dir))
        if os.path.exists(scf_work_dir):
            shutil.rmtree(scf_work_dir)
        self.assertTrue(outputs['normal_end'])
        self.assertTrue(outputs['converge'])
        self.assertAlmostEqual(outputs['energy'], ref_results['energy'], places=6)

