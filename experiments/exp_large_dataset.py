from ..resplan.workload import workload_large_dataset, root_mean_squared_error, dataset_domains
from ..resplan.utils import all_subsets, prefix_workload
import numpy as np
import time


def compute_rp_sov(system, W_per_attr, att):
    """Compute sum of variances for RP+ using Theorem 8.

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


def get_per_attr_workload(domains, bases):
    """Build per-attribute workload matrices matching the basis type."""
    W_per_attr = {}
    for i, (n, b) in enumerate(zip(domains, bases)):
        if b == 'P':
            W_per_attr[i] = prefix_workload(n)
        else:  # 'I'
            W_per_attr[i] = np.eye(n)
    return W_per_attr


def loss_RMSE(dataset_list, workload_list, repeat=1, solver="cauchy"):
    for dataset in dataset_list:
        for workload in workload_list:
            print("-----------------------------------------")
            print(dataset, workload)
            time_ls = []
            loss_ls = []
            for r in range(repeat):
                start = time.time()
                system, num_query = workload_large_dataset(dataset, workload, choice="sumvar")
                sum_var = system.get_noise_level(solver=solver)
                rmse = root_mean_squared_error(sum_var, num_query, pcost=1)
                end = time.time()

                time_ls.append(end-start)
                loss_ls.append(rmse)

            mean_time = np.mean(time_ls)
            std_time = np.std(time_ls)
            mean_loss = np.mean(loss_ls)
            std_loss = np.std(loss_ls)
            print("Time: ", mean_time, 2*std_time)
            print("Loss (identity): ", mean_loss, 2*std_loss)


def loss_RMSE_thm8(dataset_list, workload_list, repeat=1, solver="cauchy"):
    """Compute workload-aware RMSE using Theorem 8 (compute_rp_sov)."""
    for dataset in dataset_list:
        domains, bases = dataset_domains(dataset)
        W_per_attr = get_per_attr_workload(domains, bases)

        for workload in workload_list:
            print("-----------------------------------------")
            print(dataset, workload)
            time_ls = []
            id_loss_ls = []
            wkld_loss_ls = []
            for r in range(repeat):
                start = time.time()
                system, num_query = workload_large_dataset(dataset, workload, choice="sumvar")
                sum_var = system.get_noise_level(solver=solver)
                if sum_var == 0:
                    print("  solver failed, skipping")
                    continue
                id_rmse = root_mean_squared_error(sum_var, num_query, pcost=1)

                # Theorem 8: workload-aware RMSE
                total_sov = 0.0
                total_wkld_queries = 0
                for att in system.marg_dict.keys():
                    sov = compute_rp_sov(system, W_per_attr, att)
                    total_sov += sov
                    nq = 1
                    for i in att:
                        nq *= W_per_attr[i].shape[0]
                    total_wkld_queries += nq
                wkld_rmse = np.sqrt(total_sov / total_wkld_queries) if total_wkld_queries > 0 else 0
                end = time.time()

                time_ls.append(end - start)
                id_loss_ls.append(id_rmse)
                wkld_loss_ls.append(wkld_rmse)

            if not time_ls:
                print("  all repeats failed")
                continue
            mean_time = np.mean(time_ls)
            std_time = np.std(time_ls)
            mean_id = np.mean(id_loss_ls)
            mean_wkld = np.mean(wkld_loss_ls)
            print("Time: ", mean_time, 2*std_time)
            print("Loss (identity): ", mean_id)
            print("Loss (workload): ", mean_wkld)


def loss_MaxVar(dataset_list, workload_list, repeat=1, solver="cvxpy", time_limit=10):
    for dataset in dataset_list:
        for workload in workload_list:
            print("-----------------------------------------")
            print(dataset, workload)
            time_ls = []
            loss_ls = []
            if dataset == "Loans" and workload == 5:
                time_limit = 30

            for r in range(repeat):
                start = time.time()
                system, num_query = workload_large_dataset(dataset, workload, choice="maxvar")
                max_var = system.get_noise_level()
                end = time.time()

                time_ls.append(end-start)
                loss_ls.append(max_var)

            mean_time = np.mean(time_ls)
            std_time = np.std(time_ls)
            mean_loss = np.mean(loss_ls)
            std_loss = np.std(loss_ls)
            print("Time: ", mean_time, 2*std_time)
            print("Loss: ", mean_loss, 2*std_loss)


if __name__ == '__main__':
    dataset_list = ["CPS", "Loans", "Adult"]
    workload_list = [1, 2, 3, "3D"]

    print("========== Identity RMSE (Cauchy) ==========")
    loss_RMSE(dataset_list, workload_list, repeat=1, solver="cauchy")

    print("\n========== Identity RMSE (CVXPY) ==========")
    loss_RMSE(dataset_list, workload_list, repeat=1, solver="cvxpy")

    print("\n========== Theorem 8 Workload RMSE (Cauchy) ==========")
    loss_RMSE_thm8(dataset_list, workload_list, repeat=1, solver="cauchy")

    print("\n========== Theorem 8 Workload RMSE (CVXPY) ==========")
    loss_RMSE_thm8(dataset_list, workload_list, repeat=1, solver="cvxpy")

    #loss_MaxVar(dataset_list, workload_list, repeat=1, solver="cvxpy")
    #loss_MaxVar(dataset_list, workload_list, repeat=1, solver="gurobi", time_limit=10)
