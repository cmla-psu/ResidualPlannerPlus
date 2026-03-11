"""
Unit tests for Algorithm 4, Theorem 7, Theorem 8, and Monte Carlo verification.

Tests verify:
  1. Noiseless reconstruction recovers exact marginal answers
  2. Theorem 7 factored privacy cost matches explicit Kronecker construction
  3. Theorem 8 factored trace matches explicit covariance trace
  4. Monte Carlo empirical trace converges to theoretical formula
"""

import sys
import os
import unittest
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from resplan.ResPlan import ResPlanSum
from resplan.utils import (
    find_residual_basis_sum,
    subtract_matrix,
    prefix_workload,
    range_workload,
    mult_kron_vec,
    all_subsets,
)


# ---------------------------------------------------------------------------
# Class 1: Noiseless Reconstruction
# ---------------------------------------------------------------------------
class TestNoiselessReconstruction(unittest.TestCase):
    """With zero noise, reconstructed workload answers must match true answers."""

    def _run_noiseless_reconstruction(self, domains, bases, max_k=None):
        if max_k is None:
            max_k = len(domains)

        system = ResPlanSum(domains, bases)

        # Random test data
        np.random.seed(7)
        n_samples = 500
        data = np.column_stack(
            [np.random.randint(0, d, size=n_samples) for d in domains]
        )
        col_names = [f"a{i}" for i in range(len(domains))]
        df = pd.DataFrame(data, columns=col_names)
        system.input_data(df, col_names)

        # Register all k-way marginals up to max_k
        att = tuple(range(len(domains)))
        for k in range(1, max_k + 1):
            for subset in itertools.combinations(att, k):
                system.input_mech(subset)

        system.get_zero_noise_level()
        system.measurement()
        system.reconstruction(debug=True)

        for att_key, mech in system.marg_dict.items():
            noisy = mech.get_noisy_answer_vector()
            non_noisy = mech.get_non_noisy_vector()
            np.testing.assert_allclose(
                noisy,
                non_noisy,
                atol=1e-8,
                err_msg=f"Reconstruction mismatch for marginal {att_key} "
                        f"(domains={domains}, bases={bases})",
            )

    # --- individual test methods ---
    def test_II_3_4(self):
        self._run_noiseless_reconstruction([3, 4], ['I', 'I'])

    def test_PP_5_4(self):
        self._run_noiseless_reconstruction([5, 4], ['P', 'P'])

    def test_PI_4_3(self):
        self._run_noiseless_reconstruction([4, 3], ['P', 'I'])

    def test_IP_3_5(self):
        self._run_noiseless_reconstruction([3, 5], ['I', 'P'])

    def test_III_2_3_4(self):
        self._run_noiseless_reconstruction([2, 3, 4], ['I', 'I', 'I'])

    def test_PPP_4_3_5(self):
        self._run_noiseless_reconstruction([4, 3, 5], ['P', 'P', 'P'])

    def test_PIP_4_3_5(self):
        self._run_noiseless_reconstruction([4, 3, 5], ['P', 'I', 'P'])

    def test_IPI_3_2_4_maxk3(self):
        self._run_noiseless_reconstruction([3, 2, 4], ['I', 'P', 'I'], max_k=3)

    def test_PI_10_8(self):
        self._run_noiseless_reconstruction([10, 8], ['P', 'I'])

    def test_P_6(self):
        self._run_noiseless_reconstruction([6], ['P'])

    def test_I_5(self):
        self._run_noiseless_reconstruction([5], ['I'])


# ---------------------------------------------------------------------------
# Class 2: Theorem 7 — Privacy Cost
# ---------------------------------------------------------------------------
class TestPrivacyCostTheorem7(unittest.TestCase):
    """Factored pcost = prod(beta_i) must equal explicit max-diag formula."""

    def _verify_pcost(self, domains, bases, att_subset):
        system = ResPlanSum(domains, bases)

        # Factored: product of per-attribute beta_i
        beta_product = np.prod([system.pcost_res[i] for i in att_subset])

        # Explicit Kronecker construction
        R_list = [system.residual_matrix[domains[i]] for i in att_subset]
        R_A = R_list[0]
        for R in R_list[1:]:
            R_A = np.kron(R_A, R)

        Sigma_parts = [
            system.gamma_matrix[domains[i]] @ system.gamma_matrix[domains[i]].T
            for i in att_subset
        ]
        Sigma_A = Sigma_parts[0]
        for S in Sigma_parts[1:]:
            Sigma_A = np.kron(Sigma_A, S)

        explicit_pcost = np.max(np.diag(R_A.T @ np.linalg.inv(Sigma_A) @ R_A))

        self.assertTrue(
            np.isclose(explicit_pcost, beta_product, rtol=1e-6),
            f"pcost mismatch: explicit={explicit_pcost}, factored={beta_product} "
            f"(domains={domains}, bases={bases}, subset={att_subset})",
        )

    # 1-way tests
    def test_1way_I_5(self):
        self._verify_pcost([5], ['I'], (0,))

    def test_1way_P_5(self):
        self._verify_pcost([5], ['P'], (0,))

    def test_1way_P_10(self):
        self._verify_pcost([10], ['P'], (0,))

    # 2-way tests
    def test_2way_II_3_4(self):
        self._verify_pcost([3, 4], ['I', 'I'], (0, 1))

    def test_2way_PP_5_4(self):
        self._verify_pcost([5, 4], ['P', 'P'], (0, 1))

    def test_2way_PI_4_3(self):
        self._verify_pcost([4, 3], ['P', 'I'], (0, 1))

    def test_2way_IP_3_5(self):
        self._verify_pcost([3, 5], ['I', 'P'], (0, 1))

    # 3-way tests
    def test_3way_III_2_3_4(self):
        self._verify_pcost([2, 3, 4], ['I', 'I', 'I'], (0, 1, 2))

    def test_3way_PPP_4_3_5(self):
        self._verify_pcost([4, 3, 5], ['P', 'P', 'P'], (0, 1, 2))

    def test_3way_PIP_4_3_5(self):
        self._verify_pcost([4, 3, 5], ['P', 'I', 'P'], (0, 1, 2))

    # Partial subset of 3 attrs
    def test_partial_subset_02_of_3(self):
        self._verify_pcost([4, 3, 5], ['P', 'I', 'P'], (0, 2))

    # Single attr of 3 attrs
    def test_single_attr_1_of_3(self):
        self._verify_pcost([4, 3, 5], ['P', 'I', 'P'], (1,))


# ---------------------------------------------------------------------------
# Class 3: Theorem 8 — Trace Formula
# ---------------------------------------------------------------------------
class TestTraceTheorem8(unittest.TestCase):
    """Explicit covariance trace must match factored Theorem 8 formula."""

    def _verify_trace(self, domains, bases, target_att, max_k=None):
        if max_k is None:
            max_k = len(domains)

        system = ResPlanSum(domains, bases)
        att = tuple(range(len(domains)))

        # Register marginals and optimize noise levels
        for k in range(1, max_k + 1):
            for subset in itertools.combinations(att, k):
                system.input_mech(subset)
        system.get_noise_level()

        # Build full workload matrix W = kron(W_i) for target_att
        W_parts = []
        for i in target_att:
            if bases[i] == 'I':
                W_parts.append(np.eye(domains[i]))
            elif bases[i] == 'P':
                W_parts.append(prefix_workload(domains[i]))
            elif bases[i] == 'R':
                W_parts.append(range_workload(domains[i]))
            else:
                W_parts.append(np.eye(domains[i]))
        W = W_parts[0]
        for w in W_parts[1:]:
            W = np.kron(W, w)

        # ---- Explicit covariance trace ----
        total_cov = np.zeros((W.shape[0], W.shape[0]))
        for subset in all_subsets(target_att):
            sigma2 = system.res_dict[subset].noise_level

            recon_parts = []
            gamma_parts = []
            for i in target_att:
                if i in subset:
                    recon_parts.append(
                        np.linalg.pinv(system.residual_matrix[domains[i]])
                    )
                    gamma_parts.append(system.gamma_matrix[domains[i]])
                else:
                    recon_parts.append(np.ones((domains[i], 1)) / domains[i])

            recon = recon_parts[0]
            for r in recon_parts[1:]:
                recon = np.kron(recon, r)

            if gamma_parts:
                gamma_full = gamma_parts[0]
                for g in gamma_parts[1:]:
                    gamma_full = np.kron(gamma_full, g)
                M = W @ recon @ gamma_full
            else:
                # Empty subset: no gamma matrices
                M = W @ recon

            total_cov += sigma2 * (M @ M.T)

        explicit_trace = np.trace(total_cov)

        # ---- Formula trace (Theorem 8) ----
        formula_trace = 0.0
        for subset in all_subsets(target_att):
            sigma2 = system.res_dict[subset].noise_level
            prod_res = (
                np.prod([system.var_res[i] for i in subset]) if subset else 1.0
            )
            prod_sum = 1.0
            for i in target_att:
                if i not in subset:
                    prod_sum *= system.var_sum[i]
            formula_trace += sigma2 * prod_res * prod_sum

        self.assertTrue(
            np.isclose(explicit_trace, formula_trace, rtol=1e-6),
            f"Trace mismatch: explicit={explicit_trace}, formula={formula_trace} "
            f"(domains={domains}, bases={bases}, target={target_att})",
        )

    # 1-way
    def test_1way_I_5(self):
        self._verify_trace([5], ['I'], (0,))

    def test_1way_P_5(self):
        self._verify_trace([5], ['P'], (0,))

    def test_1way_P_10(self):
        self._verify_trace([10], ['P'], (0,))

    # 2-way
    def test_2way_II_3_4(self):
        self._verify_trace([3, 4], ['I', 'I'], (0, 1))

    def test_2way_PP_5_4(self):
        self._verify_trace([5, 4], ['P', 'P'], (0, 1))

    def test_2way_PI_4_3(self):
        self._verify_trace([4, 3], ['P', 'I'], (0, 1))

    def test_2way_IP_3_5(self):
        self._verify_trace([3, 5], ['I', 'P'], (0, 1))

    # 3-way
    def test_3way_III_2_3_4(self):
        self._verify_trace([2, 3, 4], ['I', 'I', 'I'], (0, 1, 2))

    def test_3way_PPP_4_3_5(self):
        self._verify_trace([4, 3, 5], ['P', 'P', 'P'], (0, 1, 2))

    def test_3way_PIP_4_3_5(self):
        self._verify_trace([4, 3, 5], ['P', 'I', 'P'], (0, 1, 2))

    # Partial subset of 3 attrs
    def test_partial_02_of_3(self):
        self._verify_trace([4, 3, 5], ['P', 'I', 'P'], (0, 2))

    # Single attr of 3 attrs
    def test_single_1_of_3(self):
        self._verify_trace([4, 3, 5], ['P', 'I', 'P'], (1,))


# ---------------------------------------------------------------------------
# Class 4: Empirical Monte Carlo Verification
# ---------------------------------------------------------------------------
class TestEmpiricalMonteCarlo(unittest.TestCase):
    """Monte Carlo empirical trace must converge to Theorem 8 formula."""

    def _run_monte_carlo(
        self, domains, bases, target_att, n_runs=5000, max_k=None, rtol=0.10
    ):
        if max_k is None:
            max_k = len(domains)

        np.random.seed(42)

        att = tuple(range(len(domains)))
        col_names = [f"a{i}" for i in range(len(domains))]
        zero_data = pd.DataFrame(
            np.zeros((0, len(domains))), columns=col_names
        )

        # --- Precompute noise levels (data-independent) ---
        ref_system = ResPlanSum(domains, bases)
        for k in range(1, max_k + 1):
            for subset in itertools.combinations(att, k):
                ref_system.input_mech(subset)
        ref_system.get_noise_level()

        noise_levels = {}
        for subset, res_mech in ref_system.res_dict.items():
            noise_levels[subset] = res_mech.noise_level

        # --- Compute formula trace for identity workload ---
        # Reconstruction outputs the marginal vector estimate (identity workload),
        # so the noise covariance uses pinv(Sub_i) @ Gamma_i for residual
        # attributes and ones(d,1)/d for sum attributes.
        formula_trace = 0.0
        for subset in all_subsets(target_att):
            sigma2 = noise_levels[subset]
            # Residual part: prod ||pinv(Sub_i) @ Gamma_i||_F^2
            prod_res = 1.0
            for i in subset:
                pinv_i = np.linalg.pinv(
                    ref_system.residual_matrix[domains[i]]
                )
                gamma_i = ref_system.gamma_matrix[domains[i]]
                prod_res *= np.linalg.norm(pinv_i @ gamma_i, "fro") ** 2
            # Sum part: prod ||ones(d,1)/d||_F^2 = prod 1/d_i
            prod_sum = 1.0
            for i in target_att:
                if i not in subset:
                    prod_sum *= 1.0 / domains[i]
            formula_trace += sigma2 * prod_res * prod_sum

        # --- Monte Carlo loop ---
        squared_norms = []
        for _ in range(n_runs):
            system = ResPlanSum(domains, bases)
            system.input_data(zero_data, col_names)
            for k in range(1, max_k + 1):
                for subset in itertools.combinations(att, k):
                    system.input_mech(subset)
            # Set precomputed noise levels
            for subset, nl in noise_levels.items():
                system.res_dict[subset].input_noise_level(nl)
            system.measurement()
            system.reconstruction(debug=False)

            noisy = system.marg_dict[target_att].get_noisy_answer_vector()
            squared_norms.append(np.sum(noisy ** 2))

        empirical_trace = np.mean(squared_norms)

        self.assertTrue(
            np.isclose(empirical_trace, formula_trace, rtol=rtol),
            f"Monte Carlo mismatch: empirical={empirical_trace:.6f}, "
            f"formula={formula_trace:.6f}, rtol={rtol} "
            f"(domains={domains}, bases={bases}, target={target_att})",
        )

    # --- individual test methods ---
    def test_mc_I_4(self):
        self._run_monte_carlo([4], ['I'], (0,))

    def test_mc_P_5(self):
        self._run_monte_carlo([5], ['P'], (0,))

    def test_mc_II_3_4(self):
        self._run_monte_carlo([3, 4], ['I', 'I'], (0, 1))

    def test_mc_PP_4_3(self):
        self._run_monte_carlo([4, 3], ['P', 'P'], (0, 1))

    def test_mc_PI_4_3(self):
        self._run_monte_carlo([4, 3], ['P', 'I'], (0, 1))

    def test_mc_PIP_3_2_4_maxk2(self):
        # With max_k=2, only up to 2-way marginals are registered.
        # Use target (0, 2) — a 2-way marginal with both P bases.
        self._run_monte_carlo(
            [3, 2, 4], ['P', 'I', 'P'], (0, 2), max_k=2
        )


if __name__ == "__main__":
    unittest.main()
