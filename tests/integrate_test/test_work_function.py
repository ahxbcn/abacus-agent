import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
import pytest
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.work_function import abacus_cal_work_function

initilize_test_env()

class TestAbacusWorkFunction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_al110 = Path(__file__).parent / 'abacus_inputs_dirs/Al110/'
        self.stru_al110 = self.abacus_inputs_dir_al110 / 'STRU'
        self.input_al110 = self.abacus_inputs_dir_al110 / 'INPUT_work_function'

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)
    
    def test_abacus_cal_work_function_al110(self):
        """
        Calculate the work function of Al(100) surface.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        
        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_al110, test_work_dir)
        shutil.copy2(self.stru_al110, test_work_dir / 'STRU')
        shutil.copy2(self.input_al110, test_work_dir / 'INPUT')

        outputs = abacus_cal_work_function(test_work_dir, vacuum_direction='y')

        print(outputs)

        self.assertIsInstance(outputs['averaged_elecstat_pot_plot'], get_path_type())
        self.assertAlmostEqual(outputs['work_function'], ref_results['work_function'], places=2)

        