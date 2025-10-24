import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.bader import abacus_badercharge_run

initilize_test_env()

class TestAbacusBaderchargeRun(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_nacl_prim = Path(__file__).parent / 'abacus_inputs_dirs/NaCl-prim'
        self.stru_bader = self.abacus_inputs_dir_nacl_prim / "STRU_bader"
        self.input_nspin1 = self.abacus_inputs_dir_nacl_prim / "INPUT_nspin1"
        self.input_nspin2 = self.abacus_inputs_dir_nacl_prim / "INPUT_nspin2"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)
    
    def test_abacus_badercharge_run_nspin1(self):
        """
        Test Bader charge calculation for nspin=1 case.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
    
        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl_prim, test_work_dir)
        shutil.copy2(self.input_nspin1, test_work_dir / "INPUT")
        shutil.copy2(self.stru_bader, test_work_dir / "STRU")
        
        outputs = abacus_badercharge_run(test_work_dir)
        print(outputs)
        abacus_workpath = outputs['abacus_workpath']
        badercharge_run_workpath = outputs['badercharge_run_workpath']
        self.assertIsInstance(abacus_workpath, get_path_type())
        self.assertIsInstance(badercharge_run_workpath, get_path_type())
        for act, ref in zip(outputs['net_charges'], ref_results['net_charges']):
            self.assertAlmostEqual(act, ref, places=3)
        for act, ref in zip(outputs['atom_labels'], ref_results['atom_labels']):
            self.assertEqual(act, ref)

    def test_abacus_badercharge_run_nspin2(self):
        """
        Test Bader charge calculation for nspin=2 case.
        """
        
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        print(ref_results)
    
        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl_prim, test_work_dir)
        shutil.copy2(self.input_nspin2, test_work_dir / "INPUT")
        shutil.copy2(self.stru_bader, test_work_dir / "STRU")
        
        outputs = abacus_badercharge_run(test_work_dir)
        print(outputs)
        abacus_workpath = outputs['abacus_workpath']
        badercharge_run_workpath = outputs['badercharge_run_workpath']
        self.assertIsInstance(abacus_workpath, get_path_type())
        self.assertIsInstance(badercharge_run_workpath, get_path_type())
        for act, ref in zip(outputs['net_charges'], ref_results['net_charges']):
            self.assertAlmostEqual(act, ref, places=3)
        for act, ref in zip(outputs['atom_labels'], ref_results['atom_labels']):
            self.assertEqual(act, ref)
