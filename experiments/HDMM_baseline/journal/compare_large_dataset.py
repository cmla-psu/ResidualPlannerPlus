"""
Compare RP+ vs HDMM on CPS/Loans/Adult with workload-aware RMSE (Theorem 8).

Uses compute_rp_sov to evaluate RP+'s actual workload RMSE, not just identity RMSE.
Applies the same small filter (< 5000) as exp_large_dataset.py.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'hdmm', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hdmm import workload, templates
from resplan.ResPlan import ResPlanSum
from resplan.utils import all_subsets, prefix_workload
from resplan.workload import dataset_domains
import numpy as np
import itertools
import time
from functools import reduce


SIZE_LIMIT = 5000


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


def get_marginals_with_filter(domains, lower, upper, size_limit=SIZE_LIMIT):
    """Return list of marginal tuples that pass the size filter."""
    num_att = len(domains)
    att = tuple(range(num_att))
    marginals = []
    for k in range(lower, upper):
        for subset in itertools.combinations(att, k):
            num_query = reduce(lambda x, y: x * y, [domains[c] for c in subset], 1)
            if 0 < num_query < size_limit:
                marginals.append(subset)
    return marginals


def get_per_attr_workload(domains, bases):
    """Build per-attribute workload matrices matching the basis type."""
    W_per_attr = {}
    for i, (n, b) in enumerate(zip(domains, bases)):
        if b == 'P':
            W_per_attr[i] = prefix_workload(n)
        else:  # 'I'
            W_per_attr[i] = np.eye(n)
    return W_per_attr


def get_hdmm_blocks(domains, bases):
    """Build HDMM workload blocks matching the basis type."""
    P = workload.Prefix
    I = workload.Identity
    blocks = []
    for n, b in zip(domains, bases):
        if b == 'P':
            blocks.append(P(n))
        else:
            blocks.append(I(n))
    return blocks


def build_hdmm_workload(blocks, marginals):
    """Build HDMM VStack workload from marginal list."""
    base = [workload.Total(W.shape[1]) for W in blocks]
    d = len(blocks)
    concat = []
    for att in marginals:
        subs = [blocks[i] if i in att else base[i] for i in range(d)]
        W = workload.Kronecker(subs)
        concat.append(W)
    if len(concat) == 0:
        return None
    return workload.VStack(concat)


def get_domain(W):
    if isinstance(W, workload.VStack):
        W = W.matrices[0]
    if isinstance(W, workload.Weighted):
        W = W.base
    return tuple(Wi.shape[1] for Wi in W.matrices)


def run_comparison(dataset, wk):
    """Run RP+ (Theorem 8) vs HDMM for a given dataset and workload config."""
    domains, bases = dataset_domains(dataset)
    num_att = len(domains)

    # Determine k-way range
    if type(wk) == int:
        lower, upper = wk, wk + 1
    elif wk == "3D":
        lower, upper = 0, 4
    else:
        raise ValueError(f"Unknown workload: {wk}")

    marginals = get_marginals_with_filter(domains, lower, upper)
    if len(marginals) == 0:
        print(f"  No marginals pass the filter for {dataset} {wk}")
        return None

    # Count total queries (workload-specific)
    W_per_attr = get_per_attr_workload(domains, bases)
    total_wkld_queries = 0
    for att in marginals:
        nq = 1
        for i in att:
            nq *= W_per_attr[i].shape[0]
        total_wkld_queries += nq

    # Also count identity queries (for comparison with exp_large_dataset)
    total_id_queries = 0
    for att in marginals:
        nq = reduce(lambda x, y: x * y, [domains[c] for c in att], 1)
        total_id_queries += nq

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}, Workload: {wk}")
    print(f"Marginals: {len(marginals)}, Workload queries: {total_wkld_queries}, Identity queries: {total_id_queries}")
    print(f"{'='*60}")

    # === RP+ ===
    t0 = time.time()
    system = ResPlanSum(domains, bases)
    for att in marginals:
        system.input_mech(att)
    obj = system.get_noise_level()  # identity-based objective
    t1 = time.time()

    # RP+ identity RMSE (same as exp_large_dataset.py)
    rp_id_rmse = np.sqrt(obj / total_id_queries)

    # RP+ workload RMSE via Theorem 8
    total_sov = 0.0
    for att in marginals:
        sov = compute_rp_sov(system, W_per_attr, att)
        total_sov += sov
    rp_wkld_rmse = np.sqrt(total_sov / total_wkld_queries)
    rp_time = t1 - t0

    print(f"RP+ identity RMSE: {rp_id_rmse:.6f}  (matches exp_large_dataset)")
    print(f"RP+ workload RMSE: {rp_wkld_rmse:.6f}  (Theorem 8)")
    print(f"RP+ time: {rp_time:.2f}s")

    # === HDMM ===
    blocks = get_hdmm_blocks(domains, bases)
    W_all = build_hdmm_workload(blocks, marginals)
    ns = get_domain(W_all)

    t2 = time.time()
    temp1 = templates.DefaultKron(ns, True)
    loss1 = temp1.optimize(W_all)
    t3 = time.time()

    temp2 = templates.DefaultUnionKron(ns, len(W_all.matrices), True)
    loss2 = temp2.optimize(W_all)
    t4 = time.time()

    temp3 = templates.Marginals(ns, True)
    loss3 = temp3.optimize(W_all)
    t5 = time.time()

    hdmm_kron_rmse = np.sqrt(loss1 / W_all.shape[0])
    hdmm_union_rmse = np.sqrt(loss2 / W_all.shape[0])
    hdmm_marg_rmse = np.sqrt(loss3 / W_all.shape[0])

    print(f"HDMM Kron  RMSE: {hdmm_kron_rmse:.6f}  ({t3 - t2:.2f}s)")
    print(f"HDMM Union RMSE: {hdmm_union_rmse:.6f}  ({t4 - t3:.2f}s)")
    print(f"HDMM Marg  RMSE: {hdmm_marg_rmse:.6f}  ({t5 - t4:.2f}s)")

    return {
        'rp_id_rmse': rp_id_rmse,
        'rp_wkld_rmse': rp_wkld_rmse,
        'rp_time': rp_time,
        'kron': hdmm_kron_rmse,
        'union': hdmm_union_rmse,
        'marg': hdmm_marg_rmse,
    }


if __name__ == '__main__':
    datasets = ["CPS", "Loans", "Adult"]
    workloads = [1, 2, 3, "3D"]

    results = {}
    for dataset in datasets:
        for wk in workloads:
            results[(dataset, wk)] = run_comparison(dataset, wk)

    # Summary
    print("\n\n" + "=" * 95)
    print("SUMMARY (small filter < 5000, original P/I bases)")
    print("=" * 95)
    print(f"{'Dataset':<8} {'Wkld':<6} {'RP+ (id)':<12} {'RP+ (Thm8)':<12} {'HDMM Kron':<12} {'HDMM Union':<12} {'HDMM Marg':<12}")
    print("-" * 95)
    for dataset in datasets:
        for wk in workloads:
            r = results[(dataset, wk)]
            if r is not None:
                print(f"{dataset:<8} {str(wk):<6} {r['rp_id_rmse']:<12.4f} {r['rp_wkld_rmse']:<12.4f} "
                      f"{r['kron']:<12.4f} {r['union']:<12.4f} {r['marg']:<12.4f}")
            else:
                print(f"{dataset:<8} {str(wk):<6} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
