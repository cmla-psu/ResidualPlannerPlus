import numpy as np
import itertools
from collections import defaultdict
from ResPlan import ResPlanMax, ResPlanSum


def test_kway(val, dim, k):
    domains = [val for _ in range(dim)]
    bases = ['I'] * dim
    system = ResPlanSum(domains, bases)

    att = tuple(range(len(domains)))
    total = 0
    subset_kway = list(itertools.combinations(att, k))
    print("num of marginals: ", len(subset_kway))
    for subset in subset_kway:
        system.input_mech(subset)
        cur_domains = [domains[c] for c in subset]
        total += np.multiply.reduce(cur_domains)
    return system, total


def test_Adult():
    domains = [100, 100, 100, 99, 85, 42, 16, 15, 9, 7, 6, 5, 2, 2]
    bases = ['P'] * 5 + ['I'] * 9
    system = ResPlanSum(domains, bases)

    att = tuple(range(len(domains)))
    total = 0
    for i in range(0, 4):
        subset_i = list(itertools.combinations(att, i))
        print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c]+0.0 for c in subset]
            total += np.multiply.reduce(cur_domains)
    print("total num of queries", total)
    return system, total


def test_Loan():
    domains = [101, 101, 101, 101, 3, 8, 36, 6, 51, 4, 5, 15]
    # domains = [3 for _ in range(24)]
    bases = ['P'] * 4 + ['I'] * 8
    system = ResPlanSum(domains, bases)

    att = tuple(range(len(domains)))
    total = 0
    for i in range(0, 4):
        subset_i = list(itertools.combinations(att, i))
        print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            cur_domains = [domains[c]+0.0 for c in subset]
            num_query = np.multiply.reduce(cur_domains)
            system.input_mech(subset)
            total += num_query
    print("total num of queries", total)
    return system, total


def test_CPS():
    domains = [50, 100, 7, 4, 2]
    bases = ['P', 'P', 'I', 'I', 'I']
    # bases = ['P'] * 5
    system = ResPlanSum(domains, bases)

    att = tuple(range(len(domains)))
    total = 0
    for i in range(0, 6):
        subset_i = list(itertools.combinations(att, i))
        print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c]+0.0 for c in subset]
            total += np.multiply.reduce(cur_domains)
    print("total num of queries", total)
    # subset = (0, 1)
    # system.input_mech(subset)
    # cur_domains = [domains[c] for c in subset]
    # total += np.multiply.reduce(cur_domains)
    return system, total


def test_synthetic(p, i):
    domains = [100] * p + [10] * i
    bases = ['P'] * p + ['I'] * i
    system = ResPlanSum(domains, bases)

    att = tuple(range(len(domains)))
    total = 0
    for i in range(0, 5):
        subset_i = list(itertools.combinations(att, i))
        print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            cur_domains = [domains[c]+0.0 for c in subset]
            num_query = np.multiply.reduce(cur_domains)
            system.input_mech(subset)
            total += num_query
    print("total num of queries", total)
    return system, total


if __name__ == "__main__":
    # system, total = test_kway(val=10, dim=10, k=2)
    # system, total = test_Adult()
    # system, total = test_CPS()
    system, total = test_Loan()
    # system, total = test_synthetic(p=5, i=0)
    obj = system.get_noise_level()
    print("obj: ", obj)
    RMSE = np.sqrt(obj / total)
    print("RMSE: ", RMSE)
