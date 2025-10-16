"""
Tests for auxillary functions in work function calculation.
"""
import unittest
import os
from pathlib import Path
import numpy as np
import tempfile
from abacusagent.modules.submodules.work_function import identify_potential_plateaus

class TestIdentifyPotentialPlateau(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)  
        self.original_cwd = os.getcwd()
        self.test_path = Path(self.test_dir.name)
        os.chdir(self.test_path)
            
    def tearDown(self):
        os.chdir(self.original_cwd) 
        
    def test_identify_potential_plateau_single_no_cross_pbc(self):
        x = np.linspace(0, np.pi*10, 1000)
        y = np.where((x < np.pi*4) | (x > np.pi*8), np.cos(x) + 1, 2)

        results = identify_potential_plateaus(y)
        results_ref = [(400, 798)]
        self.assertEqual(len(results), len(results_ref))
        for i in range(len(results)):
            self.assertEqual(results[i], results_ref[i])
    
    def test_identify_potential_plateau_single_with_cross_pbc(self):
        x = np.linspace(0, np.pi*10, 1000)
        y = np.where((x > np.pi*4) & (x < np.pi*8), np.cos(x) + 1, 2)

        results = identify_potential_plateaus(y)
        print(results)

        results_ref = [(-200, 399)]
        self.assertEqual(len(results), len(results_ref))
        for i in range(len(results)):
            self.assertEqual(results[i], results_ref[i])
    
    def test_identify_potential_plateau_multiple_no_cross_pbc(self):
        x = np.linspace(0, np.pi*10, 1000)
        y = np.full_like(x, np.cos(x) + 1)
        y[(x > np.pi*0.5) & (x < np.pi*1.5)] = 1
        y[(x > np.pi*4)   & (x < np.pi*8)  ] = 2

        results = identify_potential_plateaus(y)
        print(results)
        
        results_ref = [(51, 148), (400, 798)]
        self.assertEqual(len(results), len(results_ref))
        for i in range(len(results)):
            self.assertEqual(results[i], results_ref[i])
    
    def test_identify_potential_plateau_multiple_with_cross_pbc(self):
        x = np.linspace(0, np.pi*10, 1000)
        y = np.full_like(x, np.cos(x) + 1)
        y[(x > np.pi*4.5) & (x < np.pi*5.5)] = 1
        y[(x < np.pi*4)   | (x > np.pi*8)  ] = 2

        results = identify_potential_plateaus(y)
        print(results)
        
        results_ref = [(-200, 399), (451, 548)]
        self.assertEqual(len(results), len(results_ref))
        for i in range(len(results)):
            self.assertEqual(results[i], results_ref[i])
