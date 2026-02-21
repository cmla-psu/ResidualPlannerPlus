"""
Run RP+ with Haar tree basis ('H') and compare cell RMSE against 'R' and 'P' bases.
Domain sizes are powers of 2 (required by Haar tree).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hdmm', 'src'))

import numpy as np
import time
from resplan.ResPlan import ResPlanSum
from resplan.utils import all_subsets, range_workload, prefix_workload
from resplan.workload import root_mean_squared_error


def compute_rp_sov(system, W_per_attr, att):
    """Compute sum of variances for RP+ using Theorem 8."""
    att_subsets = all_subsets(att)
    total_sov = 0.0
    res_factor = {}
    sum_factor = {}
    for i in att:
        n_i = system.domains[i]
        W_i = W_per_attr[i]
        Sub_i = system.residual_matrix[n_i]
        Sub_pinv_i = system.residual_pinv[n_i]
        M = W_i @ Sub_pinv_i @ Sub_i
        res_factor[i] = np.linalg.norm(M, 'fro') ** 2
        ones = np.ones(n_i)
        sum_factor[i] = np.linalg.norm(W_i @ ones) ** 2 / n_i ** 2
    for subset in att_subsets:
        res_mech = system.res_dict[subset]
        sigma2 = res_mech.noise_level
        product = sigma2
        for i in att:
            if i in subset:
                product *= res_factor[i]
            else:
                product *= sum_factor[i]
        total_sov += product
    return total_sov


d = 5
n_list = [2, 4, 8, 16, 32, 64]

print(f"RP+ Cell RMSE comparison: H (Haar) vs R (Range-optimized) vs P (Prefix-optimized)")
print(f"d={d} attributes, identity workload")
print()

header = f"{'n':>4} | {'H cell':>10} {'H time':>8} | {'R cell':>10} {'R time':>8} | {'P cell':>10} {'P time':>8}"
print(header)
print("-" * len(header))

for n in n_list:
    domain = [n] * d
    att = tuple(range(d))
    results = {}

    for basis_type in ['H', 'R', 'P']:
        bases = [basis_type] * d
        t0 = time.time()
        system = ResPlanSum(domain, bases)
        system.input_mech(att)
        sum_var = system.get_noise_level()
        t1 = time.time()

        # Cell RMSE via Theorem 8
        W_cell_np = np.eye(n)
        W_per_attr_cell = {i: W_cell_np for i in range(d)}
        rp_cell_sov = compute_rp_sov(system, W_per_attr_cell, att)
        num_cells = n ** d
        cell_rmse = np.sqrt(rp_cell_sov / num_cells)

        results[basis_type] = {'cell_rmse': cell_rmse, 'time': t1 - t0}

    print(f"{n:>4} | {results['H']['cell_rmse']:>10.4f} {results['H']['time']:>7.3f}s"
          f" | {results['R']['cell_rmse']:>10.4f} {results['R']['time']:>7.3f}s"
          f" | {results['P']['cell_rmse']:>10.4f} {results['P']['time']:>7.3f}s")

print()
print()

# Also show range-query RMSE
print(f"RP+ Range Query RMSE comparison: H vs R vs P")
print(f"d={d} attributes, workload = AllRange^{d}")
print()

header2 = f"{'n':>4} | {'H range':>10} | {'R range':>10} | {'P range':>10}"
print(header2)
print("-" * len(header2))

for n in n_list:
    domain = [n] * d
    att = tuple(range(d))
    results = {}

    for basis_type in ['H', 'R', 'P']:
        bases = [basis_type] * d
        system = ResPlanSum(domain, bases)
        system.input_mech(att)
        system.get_noise_level()

        W_range_np = range_workload(n)
        W_per_attr = {i: W_range_np for i in range(d)}
        rp_range_sov = compute_rp_sov(system, W_per_attr, att)
        num_range_queries = (n * (n + 1) // 2) ** d
        range_rmse = np.sqrt(rp_range_sov / num_range_queries)

        results[basis_type] = range_rmse

    print(f"{n:>4} | {results['H']:>10.4f} | {results['R']:>10.4f} | {results['P']:>10.4f}")

print()
print()

# Prefix workload RMSE
print(f"RP+ Prefix Query RMSE comparison: H vs R vs P")
print(f"d={d} attributes, workload = Prefix^{d}")
print()

header3 = f"{'n':>4} | {'H prefix':>10} | {'R prefix':>10} | {'P prefix':>10}"
print(header3)
print("-" * len(header3))

for n in n_list:
    domain = [n] * d
    att = tuple(range(d))
    results = {}

    for basis_type in ['H', 'R', 'P']:
        bases = [basis_type] * d
        system = ResPlanSum(domain, bases)
        system.input_mech(att)
        system.get_noise_level()

        W_prefix_np = prefix_workload(n)
        W_per_attr = {i: W_prefix_np for i in range(d)}
        rp_prefix_sov = compute_rp_sov(system, W_per_attr, att)
        num_prefix_queries = n ** d
        prefix_rmse = np.sqrt(rp_prefix_sov / num_prefix_queries)

        results[basis_type] = prefix_rmse

    print(f"{n:>4} | {results['H']:>10.4f} | {results['R']:>10.4f} | {results['P']:>10.4f}")
