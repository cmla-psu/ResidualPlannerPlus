"""
Experiment: Compare ResidualPlanner+ vs HDMM on single Kronecker product workloads.

Two workload types:
  1. Range:  W = R ⊗ R ⊗ R ⊗ R ⊗ R  (AllRange^⊗5)
  2. Prefix: W = P ⊗ P ⊗ P ⊗ P ⊗ P  (Prefix^⊗5)

HDMM methods:
  - OPT_X (McKennaConvex Kron): Kronecker of 1D McKennaConvex strategies
  - OPT_M (Marginals):          marginals-based strategy

RP+ method:
  - ResPlanSum with full 5-way marginal

We report two error metrics for each method:
  - Workload RMSE: error for answering range/prefix queries
  - Cell RMSE:     error for answering cell-level queries (identity workload)

RP+ RMSE is computed via Theorem 8 from the paper.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hdmm import workload, templates
from hdmm.matrix import EkteloMatrix
from resplan.ResPlan import ResPlanSum
from resplan.utils import all_subsets, range_workload, prefix_workload
from resplan.workload import root_mean_squared_error
import numpy as np
import time


def compute_expected_error(W, A):
    """Compute trace(W^T W (A^T A)^{-1}) without calling A.sensitivity()."""
    AtA = A.gram()
    AtA1 = AtA.pinv()
    WtW = W.gram()
    if isinstance(AtA1, workload.MarginalsGram):
        WtW = workload.MarginalsGram.approximate(WtW)
    X = WtW @ AtA1
    if isinstance(X, workload.Sum):
        return sum(Y.trace() for Y in X.matrices)
    return X.trace()


def compute_rootmse(W, A):
    return np.sqrt(compute_expected_error(W, A) / W.shape[0])


def compute_rp_sov(system, W_per_attr, att):
    """
    Compute sum of variances for RP+ using Theorem 8.

    SoV(Q_A) = sum_{A' in closure(A)} sigma^2_{A'}
               * prod_{i in A'} ||W_i Sub_i^+ Sub_i||_F^2
               * prod_{j in A\\A'} ||W_j 1||^2 / n_j^2

    Args:
        system: ResPlanSum with noise levels already computed
        W_per_attr: dict {attr_index: workload_matrix} for each attribute in att
        att: tuple of attribute indices for the query
    Returns:
        total sum of variances
    """
    att_subsets = all_subsets(att)
    total_sov = 0.0

    # Precompute per-attribute factors
    res_factor = {}   # ||W_i Sub_i^+ Sub_i||_F^2 for i in A'
    sum_factor = {}   # ||W_j 1||^2 / n_j^2 for j in A \ A'
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


def run_experiment(basis_type, d=5, n_list=None):
    """
    Run the comparison for a given basis type ('R' for range, 'P' for prefix).
    """
    if n_list is None:
        n_list = [2, 4, 8, 16, 32, 64]

    if basis_type == 'R':
        basis_label = 'AllRange'
        hdmm_workload_fn = lambda n: workload.AllRange(n)
        rp_workload_fn = lambda n: range_workload(n)
        num_queries_fn = lambda n: (n * (n + 1) // 2) ** d
    elif basis_type == 'P':
        basis_label = 'Prefix'
        hdmm_workload_fn = lambda n: workload.Prefix(n)
        rp_workload_fn = lambda n: prefix_workload(n)
        num_queries_fn = lambda n: n ** d  # Prefix(n) is n x n
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    outfile = os.path.join(
        os.path.dirname(__file__),
        f'scalability_kron_comparison_{basis_type}.csv')

    print("=" * 120)
    print(f"Experiment: RP+ vs HDMM on {basis_label}^⊗{d} (single Kronecker product)")
    print("=" * 120)
    print()

    results = {}

    with open(outfile, 'w') as f:
        f.write('n,OPT_X_time,OPT_X_wkldRMSE,OPT_X_cellRMSE,'
                'OPT_M_time,OPT_M_wkldRMSE,OPT_M_cellRMSE,'
                'RPplus_time,RPplus_wkldRMSE,RPplus_cellRMSE\n')

    for n in n_list:
        domain = [n] * d

        # ---- Workload: W_i^⊗d ----
        W1d = hdmm_workload_fn(n)
        K = workload.Kronecker([W1d] * d)
        W_wkld = workload.VStack([K])

        # ---- Cell workload: I^⊗d ----
        I = workload.Identity(n)
        W_cell = workload.Kronecker([I] * d)

        # === HDMM OPT_X: Kronecker of McKennaConvex ===
        temp1 = templates.Kronecker([templates.McKennaConvex(ni) for ni in domain])
        t0 = time.time()
        temp1.optimize(W_wkld)
        t1 = time.time()
        A1 = temp1.strategy()
        hdmm_kron_wkld_rmse = compute_rootmse(W_wkld, A1)
        hdmm_kron_cell_rmse = compute_rootmse(W_cell, A1)
        hdmm_kron_time = t1 - t0

        # === HDMM OPT_M: Marginals ===
        temp3 = templates.Marginals(domain, approx=True)
        t2 = time.time()
        temp3.optimize(W_wkld)
        t3 = time.time()
        A3 = temp3.strategy()
        hdmm_marg_wkld_rmse = compute_rootmse(W_wkld, A3)
        hdmm_marg_cell_rmse = compute_rootmse(W_cell, A3)
        hdmm_marg_time = t3 - t2

        # === RP+: ResPlanSum ===
        bases = [basis_type] * d
        t4 = time.time()
        system = ResPlanSum(domain, bases)
        system.input_mech(tuple(range(d)))  # full d-way marginal
        sum_var = system.get_noise_level()
        temp_rmse = root_mean_squared_error(sum_var, n ** d, pcost=1)
        t5 = time.time()
        rp_time = t5 - t4

        # Compute RP+ workload RMSE via Theorem 8
        W_1d_np = rp_workload_fn(n)
        W_per_attr = {i: W_1d_np for i in range(d)}
        rp_wkld_sov = compute_rp_sov(system, W_per_attr, tuple(range(d)))
        num_wkld_queries = num_queries_fn(n)
        rp_wkld_rmse = np.sqrt(rp_wkld_sov / num_wkld_queries)
        

        # Compute RP+ cell RMSE via Theorem 8
        W_cell_np = np.eye(n)
        W_per_attr_cell = {i: W_cell_np for i in range(d)}
        rp_cell_sov = compute_rp_sov(system, W_per_attr_cell, tuple(range(d)))
        num_cells = n ** d
        rp_cell_rmse = np.sqrt(rp_cell_sov / num_cells)

        print(f"n={n:>3}: OPT_X wkld={hdmm_kron_wkld_rmse:.4f} cell={hdmm_kron_cell_rmse:.4f} | "
              f"OPT_M wkld={hdmm_marg_wkld_rmse:.4f} cell={hdmm_marg_cell_rmse:.4f} | "
              f"RP+ wkld={rp_wkld_rmse:.4f} cell={rp_cell_rmse:.4f} | "
              f"RP+ temp={temp_rmse:.4f}")

        results[n] = {
            'kron_time': hdmm_kron_time, 'kron_wkld': hdmm_kron_wkld_rmse, 'kron_cell': hdmm_kron_cell_rmse,
            'marg_time': hdmm_marg_time, 'marg_wkld': hdmm_marg_wkld_rmse, 'marg_cell': hdmm_marg_cell_rmse,
            'rp_time': rp_time, 'rp_wkld': rp_wkld_rmse, 'rp_cell': rp_cell_rmse,
        }

        with open(outfile, 'a') as f:
            f.write(f'{n},{hdmm_kron_time:.6f},{hdmm_kron_wkld_rmse:.6f},{hdmm_kron_cell_rmse:.6f},'
                    f'{hdmm_marg_time:.6f},{hdmm_marg_wkld_rmse:.6f},{hdmm_marg_cell_rmse:.6f},'
                    f'{rp_time:.6f},{rp_wkld_rmse:.6f},{rp_cell_rmse:.6f},{temp_rmse:.6f}\n')

    # ===== Summary tables =====
    print()
    print("=" * 110)
    print(f"{basis_label} Query RMSE  (workload = {basis_label}^⊗{d})")
    print(f"{'n':>5} | {'OPT_X':>12} {'time':>10} | {'OPT_M':>12} {'time':>10} | {'RP+':>12} {'time':>10}")
    print("-" * 110)
    for n in n_list:
        s = results[n]
        print(f"{n:>5} | {s['kron_wkld']:>12.4f} {s['kron_time']:>8.4f}s | "
              f"{s['marg_wkld']:>12.4f} {s['marg_time']:>8.4f}s | "
              f"{s['rp_wkld']:>12.4f} {s['rp_time']:>8.4f}s")

    print()
    print(f"Cell-Level RMSE  (workload = Identity^⊗{d})")
    print(f"{'n':>5} | {'OPT_X':>12} {'time':>10} | {'OPT_M':>12} {'time':>10} | {'RP+':>12} {'time':>10}")
    print("-" * 110)
    for n in n_list:
        s = results[n]
        print(f"{n:>5} | {s['kron_cell']:>12.4f} {s['kron_time']:>8.4f}s | "
              f"{s['marg_cell']:>12.4f} {s['marg_time']:>8.4f}s | "
              f"{s['rp_cell']:>12.4f} {s['rp_time']:>8.4f}s")

    print("-" * 110)
    print(f"\nResults saved to {outfile}")
    return results


if __name__ == '__main__':
    d = 5
    n_list = [2, 4, 8, 16, 32, 64]

    # --- Range workload: R^⊗5 ---
    results_R = run_experiment('R', d=d, n_list=n_list)

    print("\n\n")

    # --- Prefix workload: P^⊗5 ---
    results_P = run_experiment('P', d=d, n_list=n_list)
