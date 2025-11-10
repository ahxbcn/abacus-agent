import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
import pytest
from abacusagent.modules.tool_wrapper import *
from utils import initilize_test_env, load_test_ref_result, get_path_type

initilize_test_env()

class TestToolWrapper(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_si_prim = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim/'
        self.stru_scf = self.abacus_inputs_dir_si_prim / "STRU_scf"
        self.abacus_inputs_dir_si_prim_elastic = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim-elastic/'
        self.stru_elastic = self.abacus_inputs_dir_si_prim_elastic / "STRU_cell_relaxed"
        self.abacus_inputs_dir_fe_bcc_prim = Path(__file__).parent / 'abacus_inputs_dirs/Fe-BCC-prim/'
        self.stru_fe_bcc_prim = self.abacus_inputs_dir_fe_bcc_prim / "STRU_cell_relaxed"
        self.abacus_inputs_dir_al110 = Path(__file__).parent / 'abacus_inputs_dirs/Al110/'
        self.stru_al110 = self.abacus_inputs_dir_al110 / "STRU"
        self.abacus_inputs_dir_tial = Path(__file__).parent / 'abacus_inputs_dirs/gamma-TiAl-P4mmm/'
        self.stru_tial = self.abacus_inputs_dir_tial / "STRU"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)


    def tearDown(self):
        os.chdir(self.original_cwd)
    
    def test_run_abacus_calculation_scf(self):
        """
        Test the wrapper function of doing SCF calculation.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")

        outputs = abacus_calculation_scf(test_work_dir / "STRU",
                                         stru_type='abacus/stru',
                                         lcao=True,
                                         nspin=1,
                                         dft_functional='PBE',
                                         dftu=False,
                                         dftu_param=None,
                                         init_mag=None)
        print(outputs)

        scf_work_dir = Path(outputs['scf_work_dir']).absolute()
        self.assertIsInstance(scf_work_dir, get_path_type())
        self.assertTrue(os.path.exists(scf_work_dir))
        self.assertTrue(outputs['normal_end'])
        self.assertTrue(outputs['converge'])
        self.assertAlmostEqual(outputs['energy'], ref_results['energy'], delta=1e-6)
    
    def test_run_abacus_calculation_relax(self):
        """
        Test the wrapper function of doing relax calculation.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")

        relax_precision='medium'
        outputs = abacus_do_relax(test_work_dir / "STRU",
                                  stru_type='abacus/stru',
                                  lcao=True,
                                  nspin=1,
                                  dft_functional='PBE',
                                  dftu=False,
                                  dftu_param=None,
                                  init_mag=None,
                                  max_steps=100,
                                  relax_cell=True,
                                  relax_precision=relax_precision,
                                  relax_method='cg',
                                  fixed_axes=None)
        print(outputs)

        relax_work_path = outputs['job_path']
        new_relax_work_path = outputs['new_abacus_inputs_dir']
        self.assertIsInstance(relax_work_path, get_path_type())
        self.assertIsInstance(new_relax_work_path, get_path_type())
        self.assertTrue(outputs['result']['normal_end'])
        self.assertTrue(outputs['result']['relax_converge'])
        self.assertAlmostEqual(outputs['result']['energies'][-1], ref_results['last_energy'], delta=1e-6)
        relax_precision_contents = get_relax_precision(relax_precision)
        self.assertTrue(outputs['result']['largest_gradient'][-1] <= relax_precision_contents['force_thr_ev'])
        self.assertTrue(outputs['result']['largest_gradient_stress'][-1] <= relax_precision_contents['stress_thr'])
    
    def test_run_abacus_calculation_dos(self):
        """
        Test the abacus_calculation_scf function to calculate DOS.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", property='dos')
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
        Test the abacus_calculation_scf function to calculate DOS after a cell-relax calculation.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", relax=True, relax_cell=True, property='dos')
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        pdos_fig_path = outputs['pdos_fig_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])
    
    def test_run_abacus_calculation_dft_functional(self):
        """
        Test the abacus_calculation_scf function to calculate DOS with different DFT functional.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", dft_functional="PBEsol", property='dos')
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        pdos_fig_path = outputs['pdos_fig_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_run_abacus_calculation_nspin_init_mag(self):
        """
        Test the abacus_calculation_scf function with nspin and initial magnetic moments setted.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_fe_bcc_prim, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", property='dos', nspin=2, init_mag={"Fe": 2.0})
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        pdos_fig_path = outputs['pdos_fig_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_run_abacus_calculation_dftu(self):
        """
        Test the abacus_calculation_scf function with DFT+U parameters setted
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_fe_bcc_prim, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", property='dos', dftu=True, dftu_param={'Fe': ['d', 1.0]})
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        pdos_fig_path = outputs['pdos_fig_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])
    
    def test_run_abacus_calculation_bader_charge(self):
        """
        Test the abacus_calculation_scf function to calculate Bader charge
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = abacus_badercharge_run(stru_file=self.stru_scf,
                                         stru_type='abacus/stru',
                                         lcao=True,
                                         nspin=1,
                                         dft_functional="PBE",
                                         dftu=False,
                                         dftu_param=None,
                                         init_mag=None,
                                         relax_cell=True,
                                         relax_precision='medium',
                                         relax_method='cg',
                                         fixed_axes=None)
        print(outputs)

        abacus_workpath = outputs['abacus_workpath']
        badercharge_run_workpath = outputs['badercharge_run_workpath']
        self.assertIsInstance(abacus_workpath, get_path_type())
        self.assertIsInstance(badercharge_run_workpath, get_path_type())
        for act, ref in zip(outputs['net_charges'], ref_results['net_charges']):
            self.assertAlmostEqual(act, ref, delta=1e-3)
        for act, ref in zip(outputs['atom_labels'], ref_results['atom_labels']):
            self.assertEqual(act, ref)
    
    def test_run_abacus_calculation_band(self):
        """
        Test the abacus_calculation_scf function to calculate band
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", property='band')
        print(outputs)

        band_calc_dir = outputs['band_calc_dir']
        band_picture = outputs['band_picture']
        self.assertIsInstance(band_calc_dir, get_path_type())
        self.assertIsInstance(band_picture, get_path_type())
        self.assertAlmostEqual(outputs['band_gap'], ref_results['band_gap'], places=4)
    
    @pytest.mark.long
    def test_run_abacus_calculation_elastic_properties(self):
        """
        Test the abacus_calculation_scf function to calculate elastic properties
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_elastic, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", property='elastic_properties')
        print(outputs)
        self.assertIsInstance(outputs['elastic_cal_dir'], get_path_type())

        # Compare calculated and reference elastic tensor
        self.assertEqual(len(outputs['elastic_tensor']), len(ref_results['elastic_tensor']))
        for elastic_tensor_output_row, elastic_tensor_ref_row in zip(outputs['elastic_tensor'], ref_results['elastic_tensor']):
            self.assertEqual(len(elastic_tensor_output_row), len(elastic_tensor_ref_row))
            for element_output, element_ref in zip(elastic_tensor_output_row, elastic_tensor_ref_row):
                self.assertAlmostEqual(element_output, element_ref, delta=0.1)

        self.assertAlmostEqual(outputs['bulk_modulus'], ref_results['bulk_modulus'], delta=0.1)
        self.assertAlmostEqual(outputs['shear_modulus'], ref_results['shear_modulus'], delta=0.1)
        self.assertAlmostEqual(outputs['young_modulus'], ref_results['young_modulus'], delta=0.1)
        self.assertAlmostEqual(outputs['poisson_ratio'], ref_results['poisson_ratio'], delta=0.01)
    
    @pytest.mark.long
    def test_run_abacus_calculation_phonon_dispersion(self):
        """
        Test the abacus_calculation_scf function to calculate phonon dispersion
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", property='phonon_dispersion', relax=True)
        print(outputs)
        
        self.assertIsInstance(outputs['phonon_work_path'], get_path_type())
        self.assertIsInstance(outputs['band_dos_plot'], get_path_type())

        self.assertAlmostEqual(outputs['entropy'], ref_results['entropy'], places=2)
        self.assertAlmostEqual(outputs['free_energy'], ref_results['free_energy'], places=2)
        self.assertAlmostEqual(outputs['max_frequency_THz'], ref_results['max_frequency_THz'], places=2)
        self.assertAlmostEqual(outputs['max_frequency_K'], ref_results['max_frequency_K'], places=2)
    
    def test_run_abacus_calculation_md(self):
        """
        Test the abacus_calculation_scf function to do AIMD calculation
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        md_nstep = 5
        outputs = run_abacus_calculation(test_work_dir / "STRU", property='md',
                                         md_type = 'nve',
                                         md_nstep = md_nstep,
                                         md_dt = 1.0,
                                         md_tfirst = 300)
        
        print(outputs)
        
        self.assertTrue(outputs['normal_end'])
        self.assertEqual(outputs['traj_frame_nums'], md_nstep+1)
        self.assertIsInstance(outputs['md_work_path'], get_path_type())
        self.assertIsInstance(outputs['md_traj_file'], get_path_type())

    def test_run_abacus_calculation_vacancy_formation_energy(self):
        """
        Test the abacus_calculation_scf function to calculate vacancy formation energy
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_tial, test_work_dir / "STRU")
        
        outputs = run_abacus_calculation(test_work_dir / "STRU", property='vacancy_formation_energy',
                                         vacancy_supercell = [1, 1, 1],
                                         vacancy_element = 'Ti',
                                         vacancy_element_index = 1)
        
        print(outputs)

        self.assertIsInstance(outputs['supercell_jobpath'], get_path_type())
        self.assertIsInstance(outputs['defect_supercell_jobpath'], get_path_type())
        self.assertIsInstance(outputs['vacancy_element_crys_jobpath'], get_path_type())
        self.assertAlmostEqual(outputs['vac_formation_energy'], ref_results['vac_formation_energy'], places=3)
    
    def test_run_abacus_calculation_work_function(self):
        """
        Test the abacus_calculation_scf function to calculate work function
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_al110, test_work_dir / "STRU")

        outputs = run_abacus_calculation(self.stru_al110,
                                         vacuum_direction='y',
                                         dipole_correction=False,
                                         property='work_function')

        print(outputs)

        self.assertIsInstance(outputs['averaged_elecstat_pot_plot'], get_path_type())
        self.assertEqual(len(outputs['work_function_results']), len(ref_results['work_function_results']))
        for i in range(len(outputs['work_function_results'])):
            self.assertAlmostEqual(outputs['work_function_results'][i]['work_function'], ref_results['work_function_results'][i]['work_function'], places=2)
        
