import os
import numpy as np
from resplan.ResPlan import test_simple_adult, ResPlanSum
import itertools

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def test_reconstruction_covariance_simple_adult():
    # Use the real data and setup from test_simple_adult
    system_v1, _ = test_simple_adult()
    # Get the 2-way marginals as in test_simple_adult
    att = tuple(range(3))
    i = 2
    subset_i = list(itertools.combinations(att, i))
    # Run with original subtract_matrix (v1)
    system_v1 = ResPlanSum([2, 2, 3], ['I', 'I', 'I'], subtract_version='v1')
    import pandas as pd
    data = pd.read_csv(os.path.join(_DATA_DIR, "simple_adult.csv"))
    col_names = ['education', 'marital', 'gender']
    system_v1.input_data(data, col_names)
    for subset in subset_i:
        system_v1.input_mech(subset)
    system_v1.get_noise_level()
    system_v1.measurement()
    debug_results_v1 = system_v1.reconstruction(debug=True)

    # Run with new subtract_matrix_v2 (v2)
    system_v2 = ResPlanSum([2, 2, 3], ['I', 'I', 'I'], subtract_version='v2')
    system_v2.input_data(data, col_names)
    for subset in subset_i:
        system_v2.input_mech(subset)
    system_v2.get_noise_level()
    system_v2.measurement()
    debug_results_v2 = system_v2.reconstruction(debug=True)

    # Pick a marginal to compare (e.g., the first 2-way marginal)
    att = tuple(sorted(debug_results_v1.keys())[0])
    covar_v1 = debug_results_v1[att]['identity_cov_result']
    covar_v2 = debug_results_v2[att]['identity_cov_result']
    print(f"Covariance (v1) for {att}:\n", covar_v1)
    print(f"Covariance (v2) for {att}:\n", covar_v2)
    assert np.allclose(covar_v1, covar_v2)
    print("Difference norm:", np.linalg.norm(np.array(covar_v1) - np.array(covar_v2)))

if __name__ == "__main__":
    test_reconstruction_covariance_simple_adult() 