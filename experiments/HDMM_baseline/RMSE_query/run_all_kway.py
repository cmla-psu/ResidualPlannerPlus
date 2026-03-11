"""Run HDMM baselines with small filter (<5000) for 1,2,3,3D on CPS, Loans, Adult."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from hdmm import workload, templates
import itertools
import numpy as np
import time
from functools import reduce


def get_blocks(dataset):
    """Return per-attribute workload blocks with original P/I bases."""
    P = workload.Prefix
    I = workload.Identity
    if dataset == "CPS":
        return [P(50), P(100), I(7), I(4), I(2)]
    elif dataset == "Loans":
        return [P(101), P(101), P(101), P(101), I(3), I(8), I(36), I(6), I(51), I(4), I(5), I(15)]
    elif dataset == "Adult":
        return [P(100), P(100), P(100), P(99), P(85), I(42), I(16), I(15), I(9), I(7), I(6), I(5), I(2), I(2)]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def SmallDimKKrons(blocks, k, size=5000):
    """Build k-way Kronecker workloads, filtering by size < threshold."""
    base = [workload.Total(W.shape[1]) for W in blocks]
    d = len(blocks)
    concat = []
    for attr in itertools.combinations(range(d), k):
        subs = [blocks[i] if i in attr else base[i] for i in range(d)]
        tmp = reduce(lambda x, y: x * y, [blocks[i].shape[1] for i in attr], 1)
        if 0 < tmp < size:
            W = workload.Kronecker(subs)
            concat.append(W)
    return concat


def build_workload(dataset, wk, size=5000):
    """Build workload matching RP+'s workload_large_dataset with small filter.

    wk: int k  --> k-way only (filtered by size)
        "3D"   --> 0-through-3-way (filtered by size)
    """
    blocks = get_blocks(dataset)
    if type(wk) == int:
        lower, upper = wk, wk + 1
    elif wk == "3D":
        lower, upper = 0, 4
    else:
        raise ValueError(f"Unknown workload: {wk}")

    W_all_dim = []
    for k in range(lower, upper):
        W_all_dim += SmallDimKKrons(blocks, k, size)

    if len(W_all_dim) == 0:
        return None
    return workload.VStack(W_all_dim)


def get_domain(W):
    if isinstance(W, workload.VStack):
        W = W.matrices[0]
    if isinstance(W, workload.Weighted):
        W = W.base
    return tuple(Wi.shape[1] for Wi in W.matrices)


def run_hdmm(dataset, wk):
    """Run HDMM optimization for a given dataset and workload."""
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}, Workload: {wk}")
    print(f"{'='*50}")

    start = time.time()
    W_all = build_workload(dataset, wk)
    if W_all is None:
        print("No marginals pass the size filter!")
        return None
    ns = get_domain(W_all)
    print(f"Domain: {ns}")
    print(f"Workload shape: {W_all.shape}")
    print(f"Num sub-workloads: {len(W_all.matrices)}")

    temp1 = templates.DefaultKron(ns, True)
    temp2 = templates.DefaultUnionKron(ns, len(W_all.matrices), True)
    temp3 = templates.Marginals(ns, True)

    loss1 = temp1.optimize(W_all)
    loss2 = temp2.optimize(W_all)
    loss3 = temp3.optimize(W_all)

    rmse_kron = np.sqrt(loss1 / W_all.shape[0])
    rmse_union = np.sqrt(loss2 / W_all.shape[0])
    rmse_marg = np.sqrt(loss3 / W_all.shape[0])

    elapsed = time.time() - start

    print(f"HDMM Kron  RMSE: {rmse_kron:.6f}")
    print(f"HDMM Union RMSE: {rmse_union:.6f}")
    print(f"HDMM Marg  RMSE: {rmse_marg:.6f}")
    print(f"Time: {elapsed:.2f}s")

    return {
        'kron': rmse_kron,
        'union': rmse_union,
        'marg': rmse_marg,
        'time': elapsed,
    }


if __name__ == '__main__':
    datasets = ["CPS", "Loans", "Adult"]
    workloads = [1, 2, 3, "3D"]

    results = {}
    for dataset in datasets:
        for wk in workloads:
            key = (dataset, wk)
            results[key] = run_hdmm(dataset, wk)

    # Summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY: HDMM RMSE (small filter < 5000, original P/I bases)")
    print("=" * 70)
    print(f"{'Dataset':<10} {'Workload':<10} {'Kron':<15} {'Union':<15} {'Marg':<15}")
    print("-" * 70)
    for dataset in datasets:
        for wk in workloads:
            r = results[(dataset, wk)]
            if r is not None:
                print(f"{dataset:<10} {str(wk):<10} {r['kron']:<15.6f} {r['union']:<15.6f} {r['marg']:<15.6f}")
            else:
                print(f"{dataset:<10} {str(wk):<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
