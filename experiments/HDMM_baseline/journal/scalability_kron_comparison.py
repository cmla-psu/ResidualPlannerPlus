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

We report workload RMSE for each method.
RP+ RMSE is computed via Theorem 8 from the paper.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'hdmm', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hdmm import workload, templates
from hdmm.matrix import EkteloMatrix
from resplan.ResPlan import ResPlanSum
from resplan.utils import all_subsets, range_workload, prefix_workload
import numpy as np
import time
import gc



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


def run_experiment(basis_type, d=5, n_list=None, num_trials=5):
    """
    Run the comparison for a given basis type ('R' for range, 'P' for prefix).
    Each configuration is run num_trials times and the RMSE/time are averaged.
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
    print(f"Experiment: RP+ vs HDMM on {basis_label}^⊗{d} (single Kronecker product, {num_trials} trials)")
    print("=" * 120)
    print()

    results = {}

    with open(outfile, 'w') as f:
        f.write('n,OPT_X_time,OPT_X_wkldRMSE,'
                'OPT_M_time,OPT_M_wkldRMSE,'
                'RPplus_time,RPplus_wkldRMSE\n')

    for n in n_list:
        domain = [n] * d

        kron_wkld_list = []
        kron_time_list = []
        marg_wkld_list = []
        marg_time_list = []
        rp_wkld_list = []
        rp_time_list = []

        for trial in range(num_trials):
            # ---- Workload: W_i^⊗d ----
            W1d = hdmm_workload_fn(n)
            W_wkld = workload.Kronecker([W1d] * d)

            # === HDMM OPT_X: ===
            temp1 = templates.DefaultKron([n]*d,)
            t0 = time.time()
            loss1 = temp1.optimize(W_wkld)
            t1 = time.time()
            kron_wkld_list.append(np.sqrt(loss1 / W_wkld.shape[0]))
            kron_time_list.append(t1 - t0)

            # === HDMM OPT_M: Marginals ===
            temp3 = templates.Marginals(domain, approx=True)
            t2 = time.time()
            loss3 = temp3.optimize(W_wkld)
            t3 = time.time()
            marg_wkld_list.append(np.sqrt(loss3 / W_wkld.shape[0]))
            marg_time_list.append(t3 - t2)

            # === RP+: ResPlanSum ===
            bases = [basis_type] * d
            t4 = time.time()
            system = ResPlanSum(domain, bases)
            system.input_mech(tuple(range(d)))  # full d-way marginal
            system.get_noise_level()
            t5 = time.time()
            rp_time_list.append(t5 - t4)

            # Compute RP+ workload RMSE via Theorem 8
            W_1d_np = rp_workload_fn(n)
            W_per_attr = {i: W_1d_np for i in range(d)}
            rp_wkld_sov = compute_rp_sov(system, W_per_attr, tuple(range(d)))
            num_wkld_queries = num_queries_fn(n)
            rp_wkld_list.append(np.sqrt(rp_wkld_sov / num_wkld_queries))

        # Average over trials
        hdmm_kron_wkld_rmse = np.mean(kron_wkld_list)
        hdmm_kron_time = np.mean(kron_time_list)
        hdmm_marg_wkld_rmse = np.mean(marg_wkld_list)
        hdmm_marg_time = np.mean(marg_time_list)
        rp_wkld_rmse = np.mean(rp_wkld_list)
        rp_time = np.mean(rp_time_list)

        print(f"n={n:>3}: OPT_X wkld={hdmm_kron_wkld_rmse:.4f} | "
              f"OPT_M wkld={hdmm_marg_wkld_rmse:.4f} | "
              f"RP+ wkld={rp_wkld_rmse:.4f}")

        results[n] = {
            'kron_time': hdmm_kron_time, 'kron_wkld': hdmm_kron_wkld_rmse,
            'marg_time': hdmm_marg_time, 'marg_wkld': hdmm_marg_wkld_rmse,
            'rp_time': rp_time, 'rp_wkld': rp_wkld_rmse,
        }

        with open(outfile, 'a') as f:
            f.write(f'{n},{hdmm_kron_time:.6f},{hdmm_kron_wkld_rmse:.6f},'
                    f'{hdmm_marg_time:.6f},{hdmm_marg_wkld_rmse:.6f},'
                    f'{rp_time:.6f},{rp_wkld_rmse:.6f}\n')

    # ===== Summary table =====
    print()
    print("=" * 110)
    print(f"{basis_label} Query RMSE  (workload = {basis_label}^⊗{d}, avg over {num_trials} trials)")
    print(f"{'n':>5} | {'OPT_X':>12} {'time':>10} | {'OPT_M':>12} {'time':>10} | {'RP+':>12} {'time':>10}")
    print("-" * 110)
    for n in n_list:
        s = results[n]
        print(f"{n:>5} | {s['kron_wkld']:>12.4f} {s['kron_time']:>8.4f}s | "
              f"{s['marg_wkld']:>12.4f} {s['marg_time']:>8.4f}s | "
              f"{s['rp_wkld']:>12.4f} {s['rp_time']:>8.4f}s")

    print("-" * 110)
    print(f"\nResults saved to {outfile}")
    return results


def run_recon_experiment(d=5, n_list=None):
    """
    Reconstruction (least-squares solve) time comparison.

    Matches the setup from scalability.py (scalability_ls_0411):
      - OPT_X (DefaultKron) optimized on AllRange^⊗d
      - OPT_M (Marginals)   optimized on DimKMarginals(ns, [0..d-2])
      - RP+   (ResPlanSum)   with full d-way marginal
    All reconstruction uses float32.
    """
    if n_list is None:
        n_list = [2, 4, 8, 16, 32, 64]

    outfile = os.path.join(
        os.path.dirname(__file__), 'scalability_ls_kron_comparison.csv')

    print("\n" + "=" * 80)
    print(f"Reconstruction Time Comparison (AllRange^⊗{d})")
    print("=" * 80 + "\n")

    with open(outfile, 'w') as f:
        f.write('n,Kronecker,Marginals,RPplus\n')

    results = {}

    for n in n_list:
        domain = [n] * d
        ns = tuple(domain)

        # Workloads (matching scalability_ls_0411)
        R = workload.AllRange(n)
        W = workload.Kronecker([R] * d)
        #W1 = workload.DimKMarginals(ns, list(range(d - 1)))

        # === OPT_X: DefaultKron, optimized on AllRange^⊗d ===
        temp1 = templates.DefaultKron([n] * d)
        temp1.optimize(W)
        A1 = temp1.strategy()
        subs = []
        for sub in A1.matrices:
            subs.append(EkteloMatrix(sub.matrix.astype(np.float32)))
        A1_f32 = workload.Kronecker(subs)
        y1 = np.zeros(A1_f32.shape[0], dtype=np.float32)
        t0 = time.time()
        A1_f32.pinv().dot(y1)
        t1 = time.time()
        kron_time = t1 - t0
        print(f'  checkpt OPT_X  n={n}')
        y1 = None
        gc.collect()

        # === OPT_M: Marginals, optimized on DimKMarginals ===
        temp3 = templates.Marginals(ns, True)
        temp3.optimize(W1)
        A3 = temp3.strategy()
        A3.weights = A3.weights.astype(np.float32)
        A3.dtype = np.float32
        y3 = np.zeros(A3.shape[0], dtype=np.float32)
        t2 = time.time()
        AtA1 = A3.gram().pinv()
        AtA1.weights = AtA1.weights.astype(np.float32)
        AtA1.dtype = np.float32
        At = A3.T
        At.dtype = np.float32
        A_recon = AtA1 @ At
        A_recon.dot(y3)
        t3 = time.time()
        marg_time = t3 - t2
        y3 = None
        gc.collect()

        # === RP+: ResPlanSum reconstruction ===
        bases = ['R'] * d
        system = ResPlanSum(domain, bases)
        system.input_mech(tuple(range(d)))
        system.get_noise_level()

        # Reconstruct: for each subset, apply Kronecker pseudoinverse
        att = tuple(range(d))
        att_subsets = all_subsets(att)
        t4 = time.time()
        for subset in att_subsets:
            matrices = []
            for i in range(d):
                n_i = system.domains[i]
                if i in subset:
                    pinv_i = system.residual_pinv[n_i].astype(np.float32)
                    matrices.append(EkteloMatrix(pinv_i))
                else:
                    ones_i = np.ones((n_i, 1), dtype=np.float32)
                    matrices.append(EkteloMatrix(ones_i))
            K_pinv = workload.Kronecker(matrices)
            y_S = np.zeros(K_pinv.shape[1], dtype=np.float32)
            K_pinv.dot(y_S)
        t5 = time.time()
        rp_time = t5 - t4
        gc.collect()

        results[n] = (kron_time, marg_time, rp_time)

        line = '%d, %.6f, %.6f, %.6f' % (n, kron_time, marg_time, rp_time)
        print(f'  {line}')
        with open(outfile, 'a') as f:
            f.write(line + '\n')

    # Summary table


    
    print()
    print(f"{'n':>5} | {'Kronecker':>12} | {'Marginals':>12} | {'RP+':>12}")
    print("-" * 55)
    for n in n_list:
        kt, mt, rt = results[n]
        print(f"{n:>5} | {kt:>10.6f}s | {mt:>10.6f}s | {rt:>10.6f}s")
    print("-" * 55)
    print(f"\nResults saved to {outfile}")
    return results


if __name__ == '__main__':
    d = 5
    n_list = [2, 4, 8, 16, 32, 64]

    # --- Range workload: R^⊗5 ---
    results_R = run_experiment('R', d=d, n_list=n_list)

    print("\n\n")

    # --- Prefix workload: P^⊗5 ---
    #results_P = run_experiment('P', d=d, n_list=n_list)

    print("\n\n")

    # --- Reconstruction time comparison (matching scalability_ls_0411 setup) ---
    #results_ls = run_recon_experiment(d=d, n_list=n_list)
