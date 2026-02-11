import unittest
import os
import numpy as np
import pandas as pd
import itertools
from resplan.ResPlan import ResPlanSum
from resplan.utils import find_residual_basis_sum, find_residual_basis_max

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class TestResPlan(unittest.TestCase):
    def setUp(self):
        # Define domains for education(3), marital(2), gender(2)
        self.domains = [3, 2, 2]
        self.col_names = ['education', 'marital', 'gender']
        # Using Prefix basis for education and Identity basis for others
        self.bases = ['P', 'I', 'I']
        self.system = ResPlanSum(self.domains, self.bases)

        # Load test data
        self.data = pd.read_csv(os.path.join(_DATA_DIR, "simple_adult.csv"))
        self.system.input_data(self.data, self.col_names)
        
        # Set up a 2-way marginal query
        att = tuple(range(len(self.domains)))
        subset_i = list(itertools.combinations(att, 2))[0]  # Get first 2-way marginal
        self.system.input_mech(subset_i)
        
        # Get noise level (set to 0 for testing)
        self.system.get_noise_level(zero=True)

    def test_measurement(self):
        """Test if measurement produces expected outputs"""
        # Perform measurement
        self.system.measurement()
        
        # Check if measurement populated the res_dict with expected values
        for att, res_mech in self.system.res_dict.items():
            # Verify noisy_answer exists and has correct shape
            self.assertIsNotNone(res_mech.noisy_answer)
            self.assertIsNotNone(res_mech.true_answer)
            
            # Since noise is set to 0, noisy_answer should equal true_answer
            np.testing.assert_array_almost_equal(
                res_mech.noisy_answer,
                res_mech.true_answer
            )
            
            # Verify trueData exists and has correct shape
            self.assertIsNotNone(res_mech.trueData)
            expected_shape = np.prod([self.domains[i] for i in att if i in att])
            self.assertEqual(res_mech.trueData.shape[0], expected_shape)

    def test_reconstruction(self):
        """Test if reconstruction produces expected outputs"""
        # Perform measurement and reconstruction
        self.system.measurement()
        self.system.reconstruction()
        
        # Check if reconstruction populated marg_dict with expected values
        for att, mech in self.system.marg_dict.items():
            # Verify answers exist
            self.assertIsNotNone(mech.noisy_answer)
            self.assertIsNotNone(mech.true_answer)
            
            # Since noise is set to 0, reconstructed noisy_answer should equal true_answer
            np.testing.assert_array_almost_equal(
                mech.noisy_answer,
                mech.true_answer
            )
            
            # Verify trueData exists
            self.assertIsNotNone(mech.trueData)
            
            # Verify mean error is close to 0 (since noise=0)
            error = self.system.get_mean_error(ord=1)
            self.assertLess(error, 1e-10)

    def test_reconstruction_multiple_bases(self):
        """Test reconstruction with different basis types (P, I, R) and dimensions"""
        # Test cases with different domain sizes and basis types
        test_cases = [
            ([3, 2], ['P', 'I']),  # Small domain, mixed basis
            ([10, 5], ['P', 'P']),  # Medium domain, all prefix
            ([4, 4, 4], ['I', 'I', 'I']),  # Multiple dimensions, all identity
            ([8, 4, 2], ['P', 'I', 'R'])  # Mixed domain sizes and bases
        ]
        
        for domains, bases in test_cases:
            system = ResPlanSum(domains, bases)
            
            # Create test data
            n_samples = 100
            test_data = np.random.randint(0, min(domains), size=(n_samples, len(domains)))
            df = pd.DataFrame(test_data, columns=[f'col{i}' for i in range(len(domains))])
            system.input_data(df, df.columns)
            
            # Set up marginal queries
            att = tuple(range(len(domains)))
            for i in range(1, len(domains) + 1):
                for subset in itertools.combinations(att, i):
                    system.input_mech(subset)
            
            # Set noise to 0 and perform measurement/reconstruction
            system.get_noise_level(zero=True)
            system.measurement()
            system.reconstruction()
            
            # Verify reconstruction error is negligible
            error = system.get_mean_error(ord=1)
            self.assertLess(error, 1e-10)

    def test_orthogonality_property(self):
        """Test if Bs @ pinv(sub_matrix) is close to zero for different cases"""
        # Test cases with different domain sizes
        test_cases = [3, 5, 10, 20]
        
        for k in test_cases:
            # Test for each basis type
            for base in ['P', 'I', 'R']:
                Bs, R, Us, Ur = find_residual_basis_sum(k, base)
                
                # Check orthogonality property
                pinv_R = np.linalg.pinv(R)
                result = Bs @ pinv_R
                
                # Result should be close to zero
                self.assertTrue(np.allclose(result, np.zeros_like(result), atol=1e-10))

    def test_theorem_6_conditions(self):
        """Test Theorem 6's B_i conditions for different situations"""
        # Test cases with different domain sizes and rank conditions
        test_cases = [
            (3, 'P'),  # Small domain, prefix basis
            (5, 'I'),  # Medium domain, identity basis
            (10, 'R'),  # Larger domain, range basis
        ]
        
        for k, base in test_cases:
            Bs, R, Us, Ur = find_residual_basis_sum(k, base)
            
            # Test condition 1: B_i should be full rank
            rank_R = np.linalg.matrix_rank(R)
            expected_rank = k - 1  # Expected rank for residual matrix
            self.assertEqual(rank_R, expected_rank)
            
            #|atti|-1/|atti|==subi.T (subisubi.T)-1 subi
            subi=R
            self.assertTrue(np.allclose(np.linalg.det(Bs)-1/np.linalg.det(Bs),subi.T @ (subi @ subi.T)-1 @ subi,atol=1e-10))
    
    def test_covariance_matrix(self):
        pass


if __name__ == '__main__':
    unittest.main() 
