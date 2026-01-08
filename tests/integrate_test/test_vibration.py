import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.vibration import abacus_vibration_analysis

initilize_test_env()

class TestAbacusVibrationAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_h2 = Path(__file__).parent / 'abacus_inputs_dirs/H2/'
        self.stru_h2_relaxed = self.abacus_inputs_dir_h2 / "STRU_relaxed"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)
    
    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_abacus_vibration_analysis_h2(self):
        """
        Test the abacus_vibration_analysis function for relaxed H2 molecule.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_h2, test_work_dir)
        shutil.copy2(self.stru_h2_relaxed, test_work_dir / "STRU")

        outputs = abacus_vibration_analysis(test_work_dir,
                                            selected_atoms = [1, 2],
                                            stepsize = 0.01,
                                            temperature=400)
        print(outputs)
        
        self.assertIsInstance(outputs['vib_analysis_work_path'], get_path_type())

        for freq_output, freq_ref in zip(outputs['frequencies'], ref_results['frequencies']):
            self.assertAlmostEqual(freq_output, freq_ref, delta=0.1)

        self.assertAlmostEqual(outputs['zero_point_energy'], ref_results['zero_point_energy'], delta=0.0001)
        self.assertEqual(len(outputs['thermo_corr'].keys()), len(ref_results['thermo_corr'].keys()))
        for temperature in outputs['thermo_corr'].keys():
            self.assertAlmostEqual(outputs['thermo_corr'][temperature]['entropy'], ref_results['thermo_corr'][temperature]['entropy'], delta=0.00001)
            self.assertAlmostEqual(outputs['thermo_corr'][temperature]['free_energy'], ref_results['thermo_corr'][temperature]['free_energy'], delta=0.001)

    def test_abacus_vibration_analysis_h2_selected(self):
        """
        Test the abacus_vibration_analysis function for relaxed H2 molecule.
        Only 1 atom are selected for vibration analysis.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_h2, test_work_dir)
        shutil.copy2(self.stru_h2_relaxed, test_work_dir / "STRU")

        outputs = abacus_vibration_analysis(test_work_dir,
                                            selected_atoms = [2],
                                            stepsize = 0.01,
                                            temperature=400)
        print(outputs)

        self.assertIsInstance(outputs['vib_analysis_work_path'], get_path_type())

        for freq_output, freq_ref in zip(outputs['frequencies'], ref_results['frequencies']):
            self.assertAlmostEqual(freq_output, freq_ref, delta=0.1)

        self.assertAlmostEqual(outputs['zero_point_energy'], ref_results['zero_point_energy'], delta=0.0001)
        self.assertEqual(len(outputs['thermo_corr'].keys()), len(ref_results['thermo_corr'].keys()))
        for temperature in outputs['thermo_corr'].keys():
            self.assertAlmostEqual(outputs['thermo_corr'][temperature]['entropy'], ref_results['thermo_corr'][temperature]['entropy'], delta=0.00001)
            self.assertAlmostEqual(outputs['thermo_corr'][temperature]['free_energy'], ref_results['thermo_corr'][temperature]['free_energy'], delta=0.001)
