import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.band import abacus_cal_band

initilize_test_env()

class TestAbacusCalBand(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_si_prim = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim/'
        self.abacus_inputs_dir_fe_bcc_prim = Path(__file__).parent / 'abacus_inputs_dirs/Fe-BCC-prim/'
        self.si_stru_band = self.abacus_inputs_dir_si_prim / "STRU_band"
        self.fe_stru_band = self.abacus_inputs_dir_fe_bcc_prim / "STRU_band"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_abacus_cal_band_pyatb_nspin1(self):
        """
        Test plot band structure in PYATB mode in nspin=1 case
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.si_stru_band, test_work_dir / "STRU")

        outputs = abacus_cal_band(test_work_dir, mode='pyatb')

        band_calc_dir = outputs['band_calc_dir']
        band_picture = outputs['band_picture']
        self.assertIsInstance(band_calc_dir, get_path_type())
        self.assertIsInstance(band_picture, get_path_type())
        self.assertAlmostEqual(outputs['band_gap'], ref_results['band_gap'], places=4)

    def test_abacus_cal_band_pyatb_nspin2(self):
        """
        Test plot band structure in PYATB mode in nspin=2 case
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_fe_bcc_prim, test_work_dir)
        shutil.copy2(self.fe_stru_band, test_work_dir / "STRU")

        outputs = abacus_cal_band(test_work_dir, mode='pyatb')

        band_calc_dir = outputs['band_calc_dir']
        band_picture = outputs['band_picture']
        self.assertIsInstance(band_calc_dir, get_path_type())
        self.assertIsInstance(band_picture, get_path_type())
        self.assertAlmostEqual(outputs['band_gap'], ref_results['band_gap'], places=4)

    def test_abacus_cal_band_nscf_nspin1(self):
        """
        Test plot band structure in NSCF mode in nspin=1 case
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.si_stru_band, test_work_dir / "STRU")

        outputs = abacus_cal_band(test_work_dir, mode='nscf')

        band_calc_dir = outputs['band_calc_dir']
        band_picture = outputs['band_picture']
        self.assertIsInstance(band_calc_dir, get_path_type())
        self.assertIsInstance(band_picture, get_path_type())
        self.assertAlmostEqual(outputs['band_gap'], ref_results['band_gap'], places=4)

    def test_abacus_cal_band_nscf_nspin2(self):
        """
        Test plot band structure in NSCF mode in nspin=1 case
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_fe_bcc_prim, test_work_dir)
        shutil.copy2(self.fe_stru_band, test_work_dir / "STRU")

        outputs = abacus_cal_band(test_work_dir, mode='nscf')

        band_calc_dir = outputs['band_calc_dir']
        band_picture = outputs['band_picture']
        self.assertIsInstance(band_calc_dir, get_path_type())
        self.assertIsInstance(band_picture, get_path_type())
        self.assertAlmostEqual(outputs['band_gap'], ref_results['band_gap'], places=4)
