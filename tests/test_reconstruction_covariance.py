import numpy as np
import pandas as pd
import itertools
from resplan.ResPlan import ResPlanSum

def test_reconstruction_covariance_compare():
    domains = [2, 2, 3]
    bases = ['I', 'I', 'I']
    # Run with original subtract_matrix (v1)
    system_v1 = ResPlanSum(domains, bases, subtract_version='v1')
    num_rows = 0
    col_names = [f'attr{i+1}' for i in range(len(domains))]
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system_v1.input_data(zero_data, col_names)
    att = tuple(range(len(domains)))
    i = 2
    subset_i = list(itertools.combinations(att, i))
    for subset in subset_i:
        system_v1.input_mech(subset)
    system_v1.get_noise_level()
    system_v1.measurement()
    debug_results_v1 = system_v1.reconstruction(debug=True)

    # Run with new subtract_matrix_v2 (v2)
    system_v2 = ResPlanSum(domains, bases, subtract_version='v2')
    system_v2.input_data(zero_data, col_names)
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
    print("Difference norm:", np.linalg.norm(covar_v1[0] - covar_v2[0]))

if __name__ == "__main__":
    test_reconstruction_covariance_compare() 