import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
import pytest
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.phonon import abacus_phonon_dispersion

initilize_test_env()

@pytest.mark.long
class TestAbacusPhononDispersion(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_si_prim = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim/'
        self.stru_si_prim_cell_relaxed = self.abacus_inputs_dir_si_prim / "STRU_cell_relaxed"
        self.abacus_inputs_dir_fe_bcc_prim = Path(__file__).parent / 'abacus_inputs_dirs/Fe-BCC-prim'
        self.stru_fe_bcc_prim_cell_relaxed = self.abacus_inputs_dir_fe_bcc_prim / "STRU_cell_relaxed"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)
    
    @pytest.mark.smoke
    def test_abacus_phonon_dispersion_nspin1(self):
        """
        Test the abacus_phonon_dispersion function for Si primitive cell to test nspin=1 case.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.stru_si_prim_cell_relaxed, test_work_dir / "STRU")

        outputs = abacus_phonon_dispersion(test_work_dir,
                                           supercell = [2, 2, 2],
                                           temperature = 300,)
        
        print(outputs)
        
        self.assertIsInstance(outputs['phonon_work_path'], get_path_type())
        self.assertIsInstance(outputs['band_dos_plot'], get_path_type())

        self.assertAlmostEqual(outputs['entropy'], ref_results['entropy'], places=2)
        self.assertAlmostEqual(outputs['free_energy'], ref_results['free_energy'], places=2)
        self.assertAlmostEqual(outputs['max_frequency_THz'], ref_results['max_frequency_THz'], places=2)
        self.assertAlmostEqual(outputs['max_frequency_K'], ref_results['max_frequency_K'], places=2)

    def test_abacus_phonon_dispersion_nspin2(self):
        """
        Test the abacus_phonon_dispersion function for BCC Fe primitive cell to test nspin=2 case.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_fe_bcc_prim, test_work_dir)
        shutil.copy2(self.stru_fe_bcc_prim_cell_relaxed, test_work_dir / "STRU")

        outputs = abacus_phonon_dispersion(test_work_dir,
                                           supercell = [2, 2, 2],
                                           temperature = 300,)
        print(outputs)
        
        self.assertIsInstance(outputs['phonon_work_path'], get_path_type())
        self.assertIsInstance(outputs['band_dos_plot'], get_path_type())

        self.assertAlmostEqual(outputs['entropy'], ref_results['entropy'], places=2)
        self.assertAlmostEqual(outputs['free_energy'], ref_results['free_energy'], places=2)
        self.assertAlmostEqual(outputs['max_frequency_THz'], ref_results['max_frequency_THz'], places=2)
        self.assertAlmostEqual(outputs['max_frequency_K'], ref_results['max_frequency_K'], places=2)

    def test_abacus_phonon_dispersion_qpath(self):
        """
        Test the abacus_phonon_dispersion function for Si primitive cell with provided qpath and high_symm_points.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.stru_si_prim_cell_relaxed, test_work_dir / "STRU")

        qpath = [['G', 'X', 'K', 'G', 'L']]
        high_symm_points = {
            'G': [0.0, 0.0, 0.0],
            'X': [0.5, 0.0, 0.5],
            'U': [0.625, 0.250, 0.625],
            'K': [0.375, 0.375, 0.750],
            'L': [0.5, 0.5, 0.5],
            'W': [0.5, 0.25, 0.75]
        }

        outputs = abacus_phonon_dispersion(test_work_dir,
                                           supercell = [2, 2, 2],
                                           temperature = 300,
                                           qpath = qpath,
                                           high_symm_points=high_symm_points)
        
        print(outputs)
        
        self.assertIsInstance(outputs['phonon_work_path'], get_path_type())
        self.assertIsInstance(outputs['band_dos_plot'], get_path_type())

        self.assertAlmostEqual(outputs['entropy'], ref_results['entropy'], places=2)
        self.assertAlmostEqual(outputs['free_energy'], ref_results['free_energy'], places=2)
        self.assertAlmostEqual(outputs['max_frequency_THz'], ref_results['max_frequency_THz'], places=2)
        self.assertAlmostEqual(outputs['max_frequency_K'], ref_results['max_frequency_K'], places=2)

