import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.relax import abacus_do_relax

initilize_test_env()

class TestAbacusDoRelax(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_si_prim = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim/'
        self.stru_relax_cell = self.abacus_inputs_dir_si_prim / "STRU_relax_cell"
        self.stru_not_relax_cell = self.abacus_inputs_dir_si_prim / "STRU_no_relax_cell"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_abacus_do_relax_relax_cell(self):
        """
        Test the abacus_do_relax function with relax_cell is True.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.stru_relax_cell, test_work_dir / "STRU")

        force_thr_ev, stress_thr = 0.01, 1.0
        outputs = abacus_do_relax(test_work_dir,
                                  force_thr_ev = force_thr_ev,
                                  stress_thr_kbar = stress_thr,
                                  max_steps = 10,
                                  relax_cell = True,
                                  relax_method = 'cg')
        relax_work_path = outputs['job_path']
        new_relax_work_path = outputs['new_abacus_inputs_dir']
        self.assertIsInstance(relax_work_path, get_path_type())
        self.assertIsInstance(new_relax_work_path, get_path_type())
        self.assertTrue(outputs['result']['normal_end'])
        self.assertTrue(outputs['result']['relax_converge'])
        self.assertTrue(outputs['result']['largest_gradient'][-1] <= force_thr_ev)
        self.assertTrue(outputs['result']['largest_gradient_stress'][-1] <= stress_thr)

        if os.path.exists(relax_work_path):
            shutil.rmtree(relax_work_path)
        
        if os.path.exists(new_relax_work_path):
            shutil.rmtree(new_relax_work_path)
        
    def test_abacus_do_relax_not_relax_cell(self):
        """
        Test the abacus_do_relax function with relax_cell is False.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.stru_not_relax_cell, test_work_dir / "STRU")

        force_thr_ev = 0.01
        outputs = abacus_do_relax(test_work_dir,
                                  force_thr_ev = force_thr_ev,
                                  max_steps = 10,
                                  relax_cell = False,
                                  relax_method = 'cg',
                                  relax_new = False)
        
        relax_work_path = outputs['job_path']
        new_relax_work_path = outputs['new_abacus_inputs_dir']
        self.assertIsInstance(relax_work_path, get_path_type())
        self.assertIsInstance(new_relax_work_path, get_path_type())
        self.assertTrue(outputs['result']['normal_end'])
        self.assertTrue(outputs['result']['relax_converge'])
        self.assertTrue(outputs['result']['largest_gradient'][-1] <= force_thr_ev)
        
        if os.path.exists(relax_work_path):
            shutil.rmtree(relax_work_path)
        
        if os.path.exists(new_relax_work_path):
            shutil.rmtree(new_relax_work_path)

