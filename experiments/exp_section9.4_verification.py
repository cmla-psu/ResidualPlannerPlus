"""
Experiment: SVD bound vs Dense SDP vs HDMM OPT_X vs RP+
on single Kronecker product workloads.

Verifies that:
  1. OPT_X (DefaultKron, approx=True) matches SVD bound
  2. Dense SDP matches SVD bound (confirms Kron factorization is optimal)
  3. RP+ >= SVD bound (RP+ cannot beat the optimal Kron strategy)

The gap between RP+ and OPT_X is inherent to the sum/residual decomposition:
RP+ separates the constant direction (rank 1) from the residual (rank n-1),
introducing a cross-term penalty of 2*sqrt(a*b) per dimension (Cauchy-Schwarz).

HDMM methods use approx=True (Gaussian/zCDP model) to match RP+.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'hdmm', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hdmm import workload, templates
from resplan.ResPlan import ResPlanSum
from resplan.utils import all_subsets, range_workload, prefix_workload
import numpy as np
import time


def svdb_1d(W):
    """SVD bound for a single workload matrix."""
    V = W.gram().dense_matrix()
    eigs = np.linalg.eigvalsh(V)
    return np.sqrt(np.maximum(0, eigs)).sum() ** 2 / V.shape[0]


def svdb_kron(W_kron):
    """SVD bound for a Kronecker product workload = product of 1D bounds."""
    return np.prod([svdb_1d(Q) for Q in W_kron.matrices])


def dense_sdp(W_kron, d, num_queries):
    """Dense SDP via McKennaConvex on the full Kron gram (small N only)."""
    g1d = W_kron.matrices[0].gram().dense_matrix()
    V = g1d.copy()
    for _ in range(d - 1):
        V = np.kron(V, g1d)
    gram_wkld = workload.ExplicitGram(V, queries=num_queries)
    temp = templates.McKennaConvex(V.shape[0])
    loss = temp.optimize(gram_wkld)
    return loss


def compute_rp_wkld_sov(system, W_per_attr, att):
    """
    Workload sum-of-variances for RP+ via Theorem 8.

    Uses the gamma matrix for noise propagation:
      SoV = sum_{A'} sigma^2_{A'}
            * prod_{i in A'} ||W_i @ pinv(R_i) @ gamma_i||_F^2
            * prod_{j not in A'} ||W_j @ 1||^2 / n_j^2
    """
    att_subsets = all_subsets(att)
    total_sov = 0.0

    res_factor = {}
    sum_factor = {}
    for i in att:
        n_i = system.domains[i]
        W_i = W_per_attr[i]
        pinv_R_i = system.residual_pinv[n_i]
        G_i = system.gamma_matrix[n_i]
        res_factor[i] = np.linalg.norm(W_i @ pinv_R_i @ G_i, 'fro') ** 2
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


def run_experiment(basis_type, d=5, n_list=None, max_dense_N=1100):
    if n_list is None:
        n_list = [2, 4, 8, 16, 32, 64]

    if basis_type == 'R':
        basis_label = 'AllRange'
        hdmm_wkld_fn = lambda n: workload.AllRange(n)
        rp_wkld_fn = lambda n: range_workload(n)
        num_queries_fn = lambda n: (n * (n + 1) // 2) ** d
    elif basis_type == 'P':
        basis_label = 'Prefix'
        hdmm_wkld_fn = lambda n: workload.Prefix(n)
        rp_wkld_fn = lambda n: prefix_workload(n)
        num_queries_fn = lambda n: n ** d
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    outfile = os.path.join(
        os.path.dirname(__file__),
        f'scalability_kron_comparison_{basis_type}.csv')

    print("=" * 100)
    print(f"{basis_label}^(x){d}  —  SVD / Dense / OPT_X / RP+")
    print("=" * 100)
    print()

    results = {}

    with open(outfile, 'w') as f:
        f.write('n,SVD_RMSE,Dense_RMSE,OPT_X_RMSE,RPplus_RMSE,RPplus_over_OPT_X\n')

    for n in n_list:
        N = n ** d
        num_q = num_queries_fn(n)

        W1d = hdmm_wkld_fn(n)
        W_kron = workload.Kronecker([W1d] * d)

        # SVD bound
        svd_rmse = np.sqrt(svdb_kron(W_kron) / num_q)

        # Dense SDP (small N only)
        if N <= max_dense_N:
            dense_rmse = np.sqrt(dense_sdp(W_kron, d, num_q) / num_q)
        else:
            dense_rmse = float('nan')

        # OPT_X: DefaultKron (approx=True)
        temp_kron = templates.DefaultKron([n] * d, approx=True)
        loss_kron = temp_kron.optimize(W_kron)
        optx_rmse = np.sqrt(loss_kron / num_q)

        # RP+: ResPlanSum
        system = ResPlanSum([n] * d, [basis_type] * d)
        system.input_mech(tuple(range(d)))
        system.get_noise_level()

        W_1d_np = rp_wkld_fn(n)
        W_per_attr = {i: W_1d_np for i in range(d)}
        rp_rmse = np.sqrt(
            compute_rp_wkld_sov(system, W_per_attr, tuple(range(d))) / num_q)

        ratio = rp_rmse / optx_rmse

        dense_str = f"{dense_rmse:.4f}" if not np.isnan(dense_rmse) else "---"
        print(f"n={n:>3}:  SVD={svd_rmse:.4f}  Dense={dense_str}  "
              f"OPT_X={optx_rmse:.4f}  RP+={rp_rmse:.4f}  "
              f"(RP+/OPT_X={ratio:.4f})")

        results[n] = dict(svd=svd_rmse, dense=dense_rmse,
                          optx=optx_rmse, rp=rp_rmse, ratio=ratio)

        with open(outfile, 'a') as f:
            f.write(f'{n},{svd_rmse:.6f},{dense_rmse},{optx_rmse:.6f},'
                    f'{rp_rmse:.6f},{ratio:.6f}\n')

    # Summary table
    print()
    hdr = (f"{'n':>5} | {'SVD':>10} | {'Dense':>10} | {'OPT_X':>10} | "
           f"{'RP+':>10} | {'RP+/OPT_X':>10}")
    print(hdr)
    print("-" * len(hdr))
    for n in n_list:
        s = results[n]
        d_str = f"{s['dense']:>10.4f}" if not np.isnan(s['dense']) else f"{'---':>10}"
        print(f"{n:>5} | {s['svd']:>10.4f} | {d_str} | {s['optx']:>10.4f} | "
              f"{s['rp']:>10.4f} | {s['ratio']:>10.4f}")
    print("-" * len(hdr))
    print(f"\nSaved to {outfile}")
    return results


if __name__ == '__main__':
    d = 5
    n_list = [2, 4, 8, 16, 32, 64]

    results_R = run_experiment('R', d=d, n_list=n_list)
    print("\n")
    results_P = run_experiment('P', d=d, n_list=n_list)
