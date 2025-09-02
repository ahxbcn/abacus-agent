import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.md import abacus_run_md

initilize_test_env()

class TestAbacusRunMd(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_h2 = Path(__file__).parent / 'abacus_inputs_dirs/H2/'
        self.stru_h2_relaxed = self.abacus_inputs_dir_h2 / "STRU_relaxed"
        self.abacus_inputs_dir_nacl = Path(__file__).parent / 'abacus_inputs_dirs/NaCl-prim'
        self.stru_nacl_md = self.abacus_inputs_dir_nacl / 'STRU_dos'
        self.input_nacl_md = self.abacus_inputs_dir_nacl / 'INPUT_nspin1'

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)
    
    def test_abacus_run_md_nve(self):
        """
        Test the abacus_run_md function running molecule dynamics in NVE ensemble.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_h2, test_work_dir)
        shutil.copy2(self.stru_h2_relaxed, test_work_dir / "STRU")

        md_nstep = 5
        outputs = abacus_run_md(test_work_dir,
                                md_type = 'nve',
                                md_nstep = md_nstep,
                                md_dt = 1.0,
                                md_tfirst = 300)
        print(outputs)
        
        self.assertTrue(outputs['normal_end'])
        self.assertEqual(outputs['traj_frame_nums'], md_nstep+1)
        self.assertIsInstance(outputs['md_work_path'], get_path_type())
        self.assertIsInstance(outputs['md_traj_file'], get_path_type())
    
    def test_abacus_run_md_nvt(self):
        """
        Test the abacus_run_md function running molecule dynamics in NVT ensemble.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl, test_work_dir)
        shutil.copy2(self.stru_nacl_md, test_work_dir / "STRU")

        md_nstep = 5
        outputs = abacus_run_md(test_work_dir,
                                md_type = 'nvt',
                                md_nstep = md_nstep,
                                md_dt = 1.0,
                                md_tfirst = 300,
                                md_thermostat = 'nhc')
        
        self.assertTrue(outputs['normal_end'])
        self.assertEqual(outputs['traj_frame_nums'], md_nstep+1)
        self.assertIsInstance(outputs['md_work_path'], get_path_type())
        self.assertIsInstance(outputs['md_traj_file'], get_path_type())
    
    def test_abacus_run_md_npt(self):
        """
        Test the abacus_run_md function running molecule dynamics in NPT ensemble.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl, test_work_dir)
        shutil.copy2(self.stru_nacl_md, test_work_dir / "STRU")

        md_nstep = 5
        outputs = abacus_run_md(test_work_dir,
                                md_type = 'npt',
                                md_nstep = md_nstep,
                                md_dt = 1.0,
                                md_tfirst = 300,
                                md_thermostat = 'nhc',
                                md_pmode = 'iso')
        
        self.assertTrue(outputs['normal_end'])
        self.assertEqual(outputs['traj_frame_nums'], md_nstep+1)
        self.assertIsInstance(outputs['md_work_path'], get_path_type())
        self.assertIsInstance(outputs['md_traj_file'], get_path_type())
    
