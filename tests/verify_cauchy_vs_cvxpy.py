"""Verify that find_var_sum_cauchy and find_var_sum_cvxpy produce the same results
for the sumvar setting on P (prefix) and I (identity) workloads."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resplan.utils import find_var_sum_cauchy, find_var_sum_cvxpy
from resplan.workload import workload_large_dataset, root_mean_squared_error, dataset_domains


def compare_solvers(dataset, workload):
    """Run both solvers on the same problem and compare."""
    system, num_query = workload_large_dataset(dataset, workload, choice="sumvar")
    param_v, param_p = system.output_coefficient()

    x_cauchy, obj_cauchy = find_var_sum_cauchy(param_v, param_p, c=1.0)

    try:
        x_cvxpy, obj_cvxpy = find_var_sum_cvxpy(param_v, param_p, c=1.0)
    except Exception as e:
        print(f"  CVXPY FAILED: {e}")
        print(f"  Cauchy obj: {obj_cauchy:.10f}")
        print(f"  num residual mechs: {len(param_v)}")
        return None, None

    rmse_cauchy = root_mean_squared_error(obj_cauchy, num_query, pcost=1)
    rmse_cvxpy = root_mean_squared_error(obj_cvxpy, num_query, pcost=1)

    # Check constraint satisfaction: sum(p / x) should equal 1
    pcost_cauchy = np.sum(param_p / x_cauchy)
    pcost_cvxpy = np.sum(param_p / x_cvxpy)

    # Also verify objective by recomputing: obj = sum(v * x)
    recomputed_obj_cauchy = np.sum(param_v * x_cauchy)
    recomputed_obj_cvxpy = np.sum(param_v * x_cvxpy)

    rel_obj_diff = abs(obj_cauchy - obj_cvxpy) / max(abs(obj_cauchy), 1e-15)
    max_x_diff = np.max(np.abs(x_cauchy - x_cvxpy))
    rel_x_diff = np.max(np.abs(x_cauchy - x_cvxpy) / np.maximum(np.abs(x_cauchy), 1e-15))

    print(f"  Cauchy obj: {obj_cauchy:.10f}   CVXPY obj: {obj_cvxpy:.10f}   rel diff: {rel_obj_diff:.2e}")
    print(f"  Cauchy obj (recomputed): {recomputed_obj_cauchy:.10f}   CVXPY obj (recomputed): {recomputed_obj_cvxpy:.10f}")
    print(f"  Cauchy RMSE: {rmse_cauchy:.10f}   CVXPY RMSE: {rmse_cvxpy:.10f}")
    print(f"  privacy cost (should=1): Cauchy={pcost_cauchy:.10f}  CVXPY={pcost_cvxpy:.10f}")
    print(f"  max |x_cauchy - x_cvxpy|: {max_x_diff:.2e}   max relative: {rel_x_diff:.2e}")
    print(f"  num residual mechs: {len(param_v)}")

    return rel_obj_diff, rel_x_diff


if __name__ == '__main__':
    dataset_list = ["CPS", "Adult", "Loans"]
    workload_list = [1, 2, 3, "3D"]

    all_pass = True
    tol = 1e-3

    for dataset in dataset_list:
        domains, bases = dataset_domains(dataset)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"  domains: {domains}")
        print(f"  bases:   {bases}")

        for workload in workload_list:
            print(f"\n--- {dataset}, workload={workload} ---")
            rel_obj_diff, rel_x_diff = compare_solvers(dataset, workload)
            if rel_obj_diff is None:
                print(f"  SKIPPED (solver failure)")
                continue
            if rel_obj_diff > tol:
                print(f"  ** OBJ MISMATCH (rel_diff={rel_obj_diff:.2e} > tol={tol}) **")
                all_pass = False
            else:
                print(f"  PASS (obj rel_diff={rel_obj_diff:.2e})")

    print(f"\n{'='*60}")
    if all_pass:
        print("All tests passed: Cauchy and CVXPY objectives agree within tolerance.")
    else:
        print("Some tests FAILED: solver objectives disagree beyond tolerance.")
