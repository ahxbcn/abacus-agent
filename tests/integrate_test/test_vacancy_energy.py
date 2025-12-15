import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
import pytest
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.vacancy import abacus_cal_vacancy_formation_energy

initilize_test_env()

@pytest.mark.long
class TestAbacusEos(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_gamma_tial = Path(__file__).parent / 'abacus_inputs_dirs/gamma-TiAl-P4mmm'
        self.stru_gamma_tial = self.abacus_inputs_dir_gamma_tial / "STRU"
        self.input_gamma_tial = self.abacus_inputs_dir_gamma_tial / "INPUT"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)
    
    def test_abacus_cal_vacancy_formation_energy_gamma_tial(self):
        """
        Test abacus_cal_vacancy_formation_energy for gamma-TiAl cubic structure
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_gamma_tial, test_work_dir)
        shutil.copy2(self.stru_gamma_tial, test_work_dir / "STRU")
        shutil.copy2(self.input_gamma_tial, test_work_dir / "INPUT")

        outputs = abacus_cal_vacancy_formation_energy(test_work_dir,
                                                      supercell=[1, 1, 1],
                                                      vacancy_index=1)
        
        print(outputs)

        self.assertTrue(outputs['supercell_job_relax_converge'])
        self.assertTrue(outputs['defect_supercell_job_relax_converge'])
        self.assertAlmostEqual(outputs['vac_formation_energy'], ref_results['vac_formation_energy'], places=2)


