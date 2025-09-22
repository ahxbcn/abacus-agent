import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
from abacusagent.modules.tool_wrapper import run_abacus_calculation
from utils import initilize_test_env, load_test_ref_result, get_path_type

initilize_test_env()

class TestToolWrapper(unittest.TestCase):
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
    
    def test_run_abacus_calculation_dos(self):
        """
        Test the abacus_calculation_scf function.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU")
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        pdos_fig_path = outputs['pdos_fig_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_run_abacus_calculation_dos_relax(self):
        """
        Test the abacus_calculation_scf function.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", relax=True, relax_cell=True)
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        pdos_fig_path = outputs['pdos_fig_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])
    
    def test_run_abacus_calculation_dos_dft_functional(self):
        """
        Test the abacus_calculation_scf function using DFT functional.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", dft_functional="PBEsol")
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        pdos_fig_path = outputs['pdos_fig_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])
