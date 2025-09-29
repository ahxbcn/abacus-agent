import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.elastic import abacus_cal_elastic

initilize_test_env()

class TestAbacusCalElastic(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_si_prim = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim/'
        self.stru_cell_relaxed = self.abacus_inputs_dir_si_prim / "STRU_cell_relaxed"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_abacus_cal_elastic_si_prim(self):
        """
        Test abacus_cal_elastic used to calculate elastic properties of Si.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.stru_cell_relaxed, test_work_dir / "STRU")

        outputs = abacus_cal_elastic(test_work_dir, kspacing=0.14)
        print(outputs)
        
        self.assertIsInstance(outputs['elastic_cal_dir'], get_path_type())

        # Compare calculated and reference elastic tensor
        self.assertEqual(len(outputs['elastic_tensor']), len(ref_results['elastic_tensor']))
        for elastic_tensor_output_row, elastic_tensor_ref_row in zip(outputs['elastic_tensor'], ref_results['elastic_tensor']):
            self.assertEqual(len(elastic_tensor_output_row), len(elastic_tensor_ref_row))
            for element_output, element_ref in zip(elastic_tensor_output_row, elastic_tensor_ref_row):
                self.assertAlmostEqual(element_output, element_ref, places=3)

        self.assertAlmostEqual(outputs['bulk_modulus'], ref_results['bulk_modulus'], places=2)
        self.assertAlmostEqual(outputs['shear_modulus'], ref_results['shear_modulus'], places=2)
        self.assertAlmostEqual(outputs['young_modulus'], ref_results['young_modulus'], places=2)
        self.assertAlmostEqual(outputs['poisson_ratio'], ref_results['poisson_ratio'], places=2)

