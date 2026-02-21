"""
Experiment: Verify whether the Haar wavelet strategy matches RP+'s RMSE
for AllRange workloads.

Key insight: RP+ uses L2/Gaussian (zCDP) privacy model internally
(pcost = max diag(R^T R) = L2^2). HDMM's Static template must use
approx=True to match this (delta = max diag(A^T A)), NOT approx=False
(delta = L1(A)^2) which gives the Laplace error formula.

Methods compared:
  1. OPT_X             — DefaultKron(approx=True), HDMM's Kronecker optimizer (L2)
  2. OPT_M             — Marginals(approx=False), HDMM's marginals optimizer
  3. Haar(HDMM)        — Haar in DefaultKron's framework via
                          templates.Kronecker([Static(Haar, approx=True)]^d)
  4. RP+               — ResPlanSum with optimized strategy (base='R')
  5. RP+(Haar)          — ResPlanSum with Haar wavelet strategy (base='H'),
                          variance coefficients recomputed for range workload

Note: Marginals cannot accept Haar because it uses a different parameterization
(weighted marginal queries M(θ)), not explicit strategy matrices.

We report workload RMSE = sqrt(loss / num_queries) for each method.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'hdmm', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hdmm import workload, templates, matrix
from hdmm.matrix import EkteloMatrix
from resplan.ResPlan import ResPlanSum
from resplan.utils import all_subsets, range_workload
import numpy as np
import time


def create_rp_haar_for_range(domain, d):
    """
    Create ResPlanSum with Haar residual matrix, but with variance coefficients
    computed for the range workload (not identity).

    base='H' in ResPlanSum sets work=identity by default. We override var_sum
    and var_res by decomposing range_workload = Us*Bs + Ur*R_haar, so the noise
    levels are optimized for range queries.
    """
    system = ResPlanSum(domain, ['H'] * d)

    # Override var_sum/var_res to use range workload decomposition
    for i, k in enumerate(system.domains):
        Bs = np.ones([1, k])
        R = system.residual_matrix[k]  # haar_tree_residual(k), set by base='H'
        work = range_workload(k)

        l_mat = np.concatenate([Bs, R]).T
        r_mat = work.T
        X = np.linalg.solve(l_mat, r_mat).T
        Us = X[:, 0].reshape(-1, 1)
        Ur = X[:, 1:]

        system.var_sum[i] = np.trace(Us @ Us.T)
        system.var_res[i] = np.linalg.norm(Ur @ R, 'fro') ** 2

    return system


def compute_rp_sov(system, W_per_attr, att):
    """
    Compute sum of variances for RP+ using Theorem 8.

    SoV(Q_A) = sum_{A' in closure(A)} sigma^2_{A'}
               * prod_{i in A'} ||W_i Sub_i^+ Sub_i||_F^2
               * prod_{j in A\\A'} ||W_j 1||^2 / n_j^2
    """
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


def run_experiment(d=5, n_list=None, num_trials=5):
    """
    Compare OPT_X, OPT_M, Haar(HDMM), RP+, and RP+(Haar) on AllRange^⊗d.
    """
    if n_list is None:
        n_list = [2, 4, 8, 16, 32, 64]

    outfile = os.path.join(
        os.path.dirname(__file__), 'scalability_haar_verification.csv')

    print("=" * 150)
    print(f"Experiment: Haar Wavelet Verification — AllRange^⊗{d} ({num_trials} trials)")
    print("=" * 150)
    print()

    results = {}

    with open(outfile, 'w') as f:
        f.write('n,OPT_X_time,OPT_X_wkldRMSE,'
                'OPT_M_time,OPT_M_wkldRMSE,'
                'HaarHDMM_time,HaarHDMM_wkldRMSE,'
                'RPplus_time,RPplus_wkldRMSE,'
                'RPplusHaar_time,RPplusHaar_wkldRMSE\n')

    for n in n_list:
        domain = [n] * d
        num_queries = (n * (n + 1) // 2) ** d

        kron_wkld_list, kron_time_list = [], []
        marg_wkld_list, marg_time_list = [], []
        haar_hdmm_wkld_list, haar_hdmm_time_list = [], []
        rp_wkld_list, rp_time_list = [], []
        rp_haar_wkld_list, rp_haar_time_list = [], []

        for trial in range(num_trials):
            W1d = workload.AllRange(n)
            W_wkld = workload.Kronecker([W1d] * d)
            W_1d_np = range_workload(n)
            W_per_attr = {i: W_1d_np for i in range(d)}

            # === OPT_X: DefaultKron (approx=True for L2/Gaussian, matching RP+) ===
            temp1 = templates.DefaultKron([n] * d, approx=True)
            t0 = time.time()
            loss1 = temp1.optimize(W_wkld)
            t1 = time.time()
            kron_wkld_list.append(np.sqrt(loss1 / num_queries))
            kron_time_list.append(t1 - t0)

            # === OPT_M: Marginals ===
            temp2 = templates.Marginals(domain, approx=False)
            t2 = time.time()
            loss2 = temp2.optimize(W_wkld)
            t3 = time.time()
            marg_wkld_list.append(np.sqrt(loss2 / num_queries))
            marg_time_list.append(t3 - t2)

            # === Haar(HDMM): Haar in DefaultKron's framework (L2) ===
            # approx=True → delta = max diag(A^T A) = L2^2, matching RP+'s model
            H_dense = EkteloMatrix(matrix.Haar(n).dense_matrix())
            haar_tmpl = templates.Kronecker(
                [templates.Static(H_dense, approx=False) for _ in range(d)])
            t4 = time.time()
            loss3 = haar_tmpl.optimize(W_wkld)
            t5 = time.time()
            haar_hdmm_wkld_list.append(np.sqrt(loss3 / num_queries))
            haar_hdmm_time_list.append(t5 - t4)

            # === RP+: ResPlanSum with optimized strategy (base='R') ===
            t6 = time.time()
            system_r = ResPlanSum(domain, ['R'] * d)
            system_r.input_mech(tuple(range(d)))
            system_r.get_noise_level()
            t7 = time.time()
            rp_time_list.append(t7 - t6)
            rp_wkld_sov = compute_rp_sov(system_r, W_per_attr, tuple(range(d)))
            rp_wkld_list.append(np.sqrt(rp_wkld_sov / num_queries))

            # === RP+(Haar): ResPlanSum with Haar strategy ===
            t8 = time.time()
            system_h = create_rp_haar_for_range(domain, d)
            system_h.input_mech(tuple(range(d)))
            system_h.get_noise_level()
            t9 = time.time()
            rp_haar_time_list.append(t9 - t8)
            rp_haar_sov = compute_rp_sov(system_h, W_per_attr, tuple(range(d)))
            rp_haar_wkld_list.append(np.sqrt(rp_haar_sov / num_queries))

        # Average over trials
        kron_rmse = np.mean(kron_wkld_list)
        kron_time = np.mean(kron_time_list)
        marg_rmse = np.mean(marg_wkld_list)
        marg_time = np.mean(marg_time_list)
        haar_hdmm_rmse = np.mean(haar_hdmm_wkld_list)
        haar_hdmm_time = np.mean(haar_hdmm_time_list)
        rp_rmse = np.mean(rp_wkld_list)
        rp_time = np.mean(rp_time_list)
        rp_haar_rmse = np.mean(rp_haar_wkld_list)
        rp_haar_time = np.mean(rp_haar_time_list)

        print(f"n={n:>3}: OPT_X={kron_rmse:.4f} | OPT_M={marg_rmse:.4f} | "
              f"Haar(HDMM)={haar_hdmm_rmse:.4f} | "
              f"RP+={rp_rmse:.4f} | RP+(Haar)={rp_haar_rmse:.4f}")

        results[n] = {
            'kron_time': kron_time, 'kron_rmse': kron_rmse,
            'marg_time': marg_time, 'marg_rmse': marg_rmse,
            'haar_hdmm_time': haar_hdmm_time, 'haar_hdmm_rmse': haar_hdmm_rmse,
            'rp_time': rp_time, 'rp_rmse': rp_rmse,
            'rp_haar_time': rp_haar_time, 'rp_haar_rmse': rp_haar_rmse,
        }

        with open(outfile, 'a') as f:
            f.write(f'{n},{kron_time:.6f},{kron_rmse:.6f},'
                    f'{marg_time:.6f},{marg_rmse:.6f},'
                    f'{haar_hdmm_time:.6f},{haar_hdmm_rmse:.6f},'
                    f'{rp_time:.6f},{rp_rmse:.6f},'
                    f'{rp_haar_time:.6f},{rp_haar_rmse:.6f}\n')

    # Summary table
    print()
    print("=" * 150)
    print(f"AllRange^⊗{d} Workload RMSE (avg over {num_trials} trials)")
    print(f"{'n':>5} | {'OPT_X':>12} {'time':>10} | {'OPT_M':>12} {'time':>10} | "
          f"{'Haar(HDMM)':>12} {'time':>10} | "
          f"{'RP+':>12} {'time':>10} | {'RP+(Haar)':>12} {'time':>10}")
    print("-" * 150)
    for n in n_list:
        s = results[n]
        print(f"{n:>5} | {s['kron_rmse']:>12.4f} {s['kron_time']:>8.4f}s | "
              f"{s['marg_rmse']:>12.4f} {s['marg_time']:>8.4f}s | "
              f"{s['haar_hdmm_rmse']:>12.4f} {s['haar_hdmm_time']:>8.4f}s | "
              f"{s['rp_rmse']:>12.4f} {s['rp_time']:>8.4f}s | "
              f"{s['rp_haar_rmse']:>12.4f} {s['rp_haar_time']:>8.4f}s")
    print("-" * 150)
    print(f"\nResults saved to {outfile}")
    return results


def run_fixed_n_experiment(d=5, n=8):
    """
    Compare all methods at fixed n=8, d=5 for AllRange workload.
    Print each method's 1D strategy matrix and RMSE.
    """
    domain = [n] * d
    num_queries = (n * (n + 1) // 2) ** d

    W1d = workload.AllRange(n)
    W_wkld = workload.Kronecker([W1d] * d)
    W_1d_np = range_workload(n)
    W_per_attr = {i: W_1d_np for i in range(d)}

    print("=" * 100)
    print(f"Fixed n={n}, d={d}, AllRange^⊗{d} workload")
    print(f"num_queries = {num_queries}")
    print("=" * 100)

    results = {}
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    # ========== 1. OPT_X (approx=True, L2/Gaussian) ==========
    temp = templates.DefaultKron([n] * d, approx=True)
    loss = temp.optimize(W_wkld)
    rmse = np.sqrt(loss / num_queries)
    # BestTemplate wraps the winner; get 1D strategy
    A_1d = temp._templates[0].strategy()
    A_1d_np = A_1d.dense_matrix()
    print(f"\n{'='*80}")
    print(f"1. OPT_X (approx=True, L2/Gaussian)")
    print(f"   RMSE = {rmse:.6f}")
    print(f"   1D strategy matrix A_i ({A_1d_np.shape[0]}x{A_1d_np.shape[1]}):")
    print(A_1d_np)
    results['OPT_X(L2)'] = rmse

    # ========== 2. OPT_X (approx=False, L1/Laplace) ==========
    temp = templates.DefaultKron([n] * d, approx=False)
    loss = temp.optimize(W_wkld)
    rmse = np.sqrt(loss / num_queries)
    A_1d = temp._templates[0].strategy()
    A_1d_np = A_1d.dense_matrix()
    print(f"\n{'='*80}")
    print(f"2. OPT_X (approx=False, L1/Laplace)")
    print(f"   RMSE = {rmse:.6f}")
    print(f"   1D strategy matrix A_i ({A_1d_np.shape[0]}x{A_1d_np.shape[1]}):")
    print(A_1d_np)
    results['OPT_X(L1)'] = rmse

    # ========== 3. OPT_M (approx=True) ==========
    temp = templates.Marginals(domain, approx=True)
    loss = temp.optimize(W_wkld)
    rmse = np.sqrt(loss / num_queries)
    strat = temp.strategy()
    weights = strat.weights
    print(f"\n{'='*80}")
    print(f"3. OPT_M (approx=True)")
    print(f"   RMSE = {rmse:.6f}")
    print(f"   Marginal weights (2^{d}={2**d} subsets):")
    for idx in range(2 ** d):
        if weights[idx] > 1e-6:
            subset = tuple(j for j in range(d) if idx & (1 << j))
            print(f"     subset {str(subset):>20s}: weight = {weights[idx]:.6f}")
    results['OPT_M(L2)'] = rmse

    # ========== 4. OPT_M (approx=False) ==========
    temp = templates.Marginals(domain, approx=False)
    loss = temp.optimize(W_wkld)
    rmse = np.sqrt(loss / num_queries)
    strat = temp.strategy()
    weights = strat.weights
    print(f"\n{'='*80}")
    print(f"4. OPT_M (approx=False)")
    print(f"   RMSE = {rmse:.6f}")
    print(f"   Marginal weights (2^{d}={2**d} subsets):")
    for idx in range(2 ** d):
        if weights[idx] > 1e-6:
            subset = tuple(j for j in range(d) if idx & (1 << j))
            print(f"     subset {str(subset):>20s}: weight = {weights[idx]:.6f}")
    results['OPT_M(L1)'] = rmse

    # ========== 5. Haar (approx=True, L2) ==========
    H_dense = EkteloMatrix(matrix.Haar(n).dense_matrix())
    H_np = H_dense.dense_matrix()
    haar_tmpl = templates.Kronecker(
        [templates.Static(H_dense, approx=True) for _ in range(d)])
    loss = haar_tmpl.optimize(W_wkld)
    rmse = np.sqrt(loss / num_queries)
    print(f"\n{'='*80}")
    print(f"5. Haar (approx=True, L2)")
    print(f"   RMSE = {rmse:.6f}")
    print(f"   Haar strategy matrix ({H_np.shape[0]}x{H_np.shape[1]}):")
    print(H_np)
    results['Haar(L2)'] = rmse

    # ========== 6. Haar (approx=False, L1) ==========
    haar_tmpl = templates.Kronecker(
        [templates.Static(H_dense, approx=False) for _ in range(d)])
    loss = haar_tmpl.optimize(W_wkld)
    rmse = np.sqrt(loss / num_queries)
    print(f"\n{'='*80}")
    print(f"6. Haar (approx=False, L1)")
    print(f"   RMSE = {rmse:.6f}")
    print(f"   Haar strategy matrix: same as above")
    results['Haar(L1)'] = rmse

    # ========== 7. RP+ (base='R', McKennaConvex) ==========
    system_r = ResPlanSum(domain, ['R'] * d)
    system_r.input_mech(tuple(range(d)))
    system_r.get_noise_level()
    rp_sov = compute_rp_sov(system_r, W_per_attr, tuple(range(d)))
    rmse = np.sqrt(rp_sov / num_queries)
    R_mat = system_r.residual_matrix[n]
    Bs = np.ones([1, n])
    full_strat_r = np.concatenate([Bs, R_mat])
    print(f"\n{'='*80}")
    print(f"7. RP+ (base='R', McKennaConvex)")
    print(f"   RMSE = {rmse:.6f}")
    print(f"   Full 1D strategy [Bs; R] ({full_strat_r.shape[0]}x{full_strat_r.shape[1]}):")
    print(full_strat_r)
    results['RP+'] = rmse

    # ========== 8. RP+(Haar) (base='H', Haar wavelet) ==========
    system_h = create_rp_haar_for_range(domain, d)
    system_h.input_mech(tuple(range(d)))
    system_h.get_noise_level()
    rp_haar_sov = compute_rp_sov(system_h, W_per_attr, tuple(range(d)))
    rmse = np.sqrt(rp_haar_sov / num_queries)
    R_haar = system_h.residual_matrix[n]
    full_strat_h = np.concatenate([Bs, R_haar])
    print(f"\n{'='*80}")
    print(f"8. RP+(Haar) (base='H', Haar wavelet)")
    print(f"   RMSE = {rmse:.6f}")
    print(f"   Full 1D strategy [Bs; R_haar] ({full_strat_h.shape[0]}x{full_strat_h.shape[1]}):")
    print(full_strat_h)
    results['RP+(Haar)'] = rmse

    # ========== Summary ==========
    print(f"\n{'='*100}")
    print(f"Summary — AllRange^⊗{d}, n={n}")
    print(f"{'Method':>20s}   {'RMSE':>12s}")
    print("-" * 40)
    for name, rmse in results.items():
        print(f"{name:>20s}   {rmse:>12.6f}")
    print("-" * 40)

    return results


if __name__ == '__main__':
    # --- Fixed n=8 experiment with strategy matrix printout ---
    results_fixed = run_fixed_n_experiment(d=5, n=8)

    # --- Scalability experiment (comment out if not needed) ---
    # d = 5
    # n_list = [2, 4, 8, 16, 32, 64]
    # results = run_experiment(d=d, n_list=n_list)
