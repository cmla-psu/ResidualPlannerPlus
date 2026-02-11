from ResPlan import ResPlanSum
import numpy as np
import itertools
import time
import pandas as pd
from workload import workload_allkway, root_mean_squared_error


def test_allkway_csv(n, d=5, k=3):
    domains = [n] * d
    bases = ['R'] * d
    col_names = [str(i) for i in range(d)]
    system = ResPlanSum(domains, bases)
    data_numpy = np.zeros([10_000, d])
    df = pd.DataFrame(data_numpy, columns=col_names)
    system.input_data(df, col_names)

    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, k+1):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way queries: ", len(subset_i))
        for t, subset in enumerate(subset_i):
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.prod(cur_domains)
            if t % 10_000 == 0 and t > 0:
                print("Selecting queries: ", t)

                
    # print("total num of queries: ", total, "\n")
    return system, total


def time_reconstruction(n_list, d_list, repeat=1):
    for n in n_list:
        for d in d_list:
            print("-------------------------------------------------------------")
            print("n = ", n, " d = ", d)
            reconstruct_time = []

            start = time.time()
            system, num_query = test_allkway_csv(n, d, 3)
            #system, num_query = workload_allkway(n, d, 3, choice="sumvar")
            sum_var = system.get_noise_level()
            time_select = time.time()
            print("time_select: ", time_select - start, "\n")

            system.measurement()
            time_measure = time.time()
            print("time_measure:", time_measure - time_select, "\n")
            for k in range(repeat):
                start = time.time()
                system.reconstruction()
                end = time.time()
                reconstruct_time.append(end - start)
                print("time_reconstruction: ", end - start, "\n")

            mean = np.mean(reconstruct_time)
            std = np.std(reconstruct_time)
            print("Time mean value and 2*std: ", mean, 2*std)


if __name__ == '__main__':
    start = time.time()

    pcost = 1
    n_list = [10]
    d_list = [2,6,10,12,14,15,20,30,50,100]
    time_reconstruction(n_list, d_list, repeat=5)

    # # system, total = test_Adult()
    # #system, total = test_allkway_csv(n=10, d=8, k=2)
    # system, total = workload_allkway(n=10, d=8, k=2, choice="maxvar")
    # obj = system.get_noise_level(zero=True)
    # print("obj: ", obj)
    # system.measurement()
    # system.reconstruction()
    # l_error = system.get_mean_error(ord=1)
    # print("Mean Error: ", l_error)

    # end = time.time()
    # print("time is: ", end-start)

