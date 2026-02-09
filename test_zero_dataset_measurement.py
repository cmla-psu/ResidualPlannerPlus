import numpy as np
import pandas as pd
import itertools
from fractions import Fraction
from ResPlan import ResPlanSum

def mult_kron_vec(mat_ls, vec):
    """Fast Kronecker matrix vector multiplication."""
    V = vec.reshape(-1, 1)
    row = 1
    X = V.T
    for Q in mat_ls[::-1]:
        m, n = Q.shape
        row *= m
        X = Q.dot(X.reshape(-1, n).T)
    return X.reshape(row, -1)


def kron_vec_np(mat_ls, vec=None):
    """Kronecker product using np.kron. If vec is provided, multiply by vec; else, return the full Kronecker matrix."""
    kron_mat = mat_ls[0]
    for Q in mat_ls[1:]:
        kron_mat = np.kron(kron_mat, Q)
    if vec is None:
        return kron_mat
    else:
        return kron_mat @ vec


def rational_approx_sqrt(x, max_denominator=100):
    """
    Find the best rational approximation s/t for sqrt(x).
    Returns (s, t) as integers.
    """
    sqrt_x = np.sqrt(x)
    frac = Fraction(sqrt_x).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator


def test_zero_dataset_measurement_sequential():
    """
    For a zero dataset, for each measurement, generate z ~ N(0, r^2 I) where
    r^2 = s^2 * |Atti|^2 for all atti in the measurement, with s/t ~ sqrt(noise_level).
    Return and average zz^T over 1000 runs for each query.
    """
    # Setup: domains, bases, and system
    domains = [2, 2, 3]
    col_names = ['attr1', 'attr2', 'attr3']
    bases = ['I', 'I', 'I']
    system = ResPlanSum(domains, bases)

    # Zero dataset
    num_rows = 0
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system.input_data(zero_data, col_names)

    # Add 2-way marginals
    #we only add 2-way marginals for now
    att = tuple(range(len(domains)))
    i = 2
    subset_i = list(itertools.combinations(att, i))
    for subset in subset_i:
        system.input_mech(subset)

    #get noise level for all residual mechanisms
    system.get_noise_level()
    
    # For each measurement, generate z ~ N(0, r^2 I) and average zz^T over 1000 runs
    n_runs = 10000
    results = {}
    for att_key, res_mech in system.res_dict.items():
        noise_level = res_mech.noise_level
        res_list = [mat / domains[i] for mat, i in zip(res_mech.res_mat_list, att_key)]
        s, t = rational_approx_sqrt(noise_level)
        # For each attribute in att_key, get |Atti|
        att_sizes = [domains[i] for i in att_key]
        if len(att_sizes) == 0:
            continue  # skip empty subset
        # r^2 = s^2 * |Atti|^2 for all atti (product)
        r2 = (s/t)**2 * np.prod([size**2 for size in att_sizes])
        col_size = np.prod(att_sizes)
        cov_sum = None
        for _ in range(n_runs):
            z = np.sqrt(r2) * np.random.normal(size=[col_size, 1])
            z1 = mult_kron_vec(res_list, z)
            cov = z1 @ z1.T
            if cov_sum is None:
                cov_sum = np.zeros_like(cov)
            cov_sum += cov
        avg_cov = cov_sum / n_runs
        results[att_key] = avg_cov
        print(f"Att {att_key}: s/t={s}/{t}, r^2={r2}, avg_cov shape={avg_cov.shape}")
        print("First few elements of avg_cov (flattened):", avg_cov.flatten()[:5])

    # Optionally, save results to file
    # np.savez('zero_dataset_measurement_covariances.npz', **results)
    return results

def test_zero_dataset_measurement_with_avg_cov():
    """
    For a zero dataset, for each measurement, call res_mech.measure with n_runs, get avg_cov, and compare to kron_vec_np(res_mech.res_mat_list) @ kron_vec_np(res_mech.res_mat_list).T
    """
    domains = [2, 2, 3]
    col_names = ['attr1', 'attr2', 'attr3']
    bases = ['I', 'I', 'I']
    system = ResPlanSum(domains, bases)

    num_rows = 0
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system.input_data(zero_data, col_names)

    att = tuple(range(len(domains)))
    i = 2
    subset_i = list(itertools.combinations(att, i))
    for subset in subset_i:
        system.input_mech(subset)

    system.get_noise_level()
    n_runs = 1000
    for att_key, res_mech in system.res_dict.items():
        att_sizes = [domains[i] for i in att_key]
        if len(att_sizes) == 0:
            continue
        avg_cov = res_mech.measure(n_runs=n_runs)
        kron_mat = kron_vec_np(res_mech.res_mat_list)
        kron_cov = kron_mat @ kron_mat.T
        print(f"Att {att_key}: avg_cov shape={avg_cov.shape}, kron_cov shape={kron_cov.shape}")
        print("First few elements of avg_cov (flattened):", avg_cov.flatten()[:5])
        print("First few elements of kron_cov (flattened):", kron_cov.flatten()[:5])
        # Unit test: avg_cov should be close to kron_cov up to scaling (statistical error)
        assert np.allclose(avg_cov / np.trace(avg_cov), kron_cov / np.trace(kron_cov), atol=0.05), f"Covariance mismatch for {att_key}"
        print(f"Covariance test passed for {att_key}\n")

if __name__ == "__main__":
    test_zero_dataset_measurement_sequential() 