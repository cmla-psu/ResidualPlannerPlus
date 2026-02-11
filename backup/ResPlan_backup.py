import numpy as np
import itertools
from collections import defaultdict
import time
from utils import *
from functools import reduce

from scipy.sparse import csr_matrix
import pandas as pd


"""
domains = [3, 3, 3 ,3 ,3]
5 attributes, each attribute 3 values
-------------------------------------------
att = ()        --> sum query
att = (0)       --> marginal A
att = (2, 3)    --> marginal CD
att = (0, 1 ,2) --> marginal ABC
"""


class MargMech:

    def __init__(self, domains, att, var_bound=1.0):
        self.domains = domains
        self.num_att = len(domains)
        self.att = att
        self.var_bound = var_bound
        self.covar = None
        self.noisy_answer = None
        cur_domains = [self.domains[at] for at in att]
        self.num_queries = np.prod([c + 0.0 for c in cur_domains])
        self.variance = None
        self.true_answer = None
        self.trueData=None
        pass

    def output_bound(self):
        return self.var_bound

    def input_noisy_answer(self, answer):
        self.noisy_answer = answer

    def input_true_answer(self, answer):
        self.true_answer = answer
    
    def input_true_data(self, data):
        self.trueData=data

    def get_true_data(self):
        return self.trueData

    def get_num_queries(self):
        return self.num_queries

    def input_variance(self, var):
        self.variance = var

    def output_variance(self):
        return self.variance

    def get_noisy_answer(self):
        return self.noisy_answer

    def get_true_answer(self):
        return self.true_answer

    pass


class ResMech:

    def __init__(self, domains, att, residual_matrix):
        self.col_names = None
        self.data = None
        self.domains = domains
        self.num_att = len(self.domains)
        self.att = att
        self.core_mat = None
        self.res_mat_list = []
        self.get_core_matrix(residual_matrix)

        self.noise_level = None
        self.covar = None
        self.calculated = False
        self.recon_answer = None
        self.noisy_answer = None
        self.true_answer = None
        self.true_recon_answer = None
        self.trueData=None

    def get_core_matrix(self, residual_matrix):
        att_set = set(list(self.att))
        for i in range(0, self.num_att):
            att_size = self.domains[i]
            if i in att_set:
                res_mat = residual_matrix[att_size]
                self.res_mat_list.append(res_mat)

    def input_noise_level(self, noise_level):
        self.noise_level = noise_level
        self.calculated = True

    def output_noise_level(self):
        if self.calculated:
            return self.noise_level
        else:
            print("Not yet calculated!")
            return 0.0

    def is_calculated(self):
        return self.calculated

    def input_data(self, data, col_names):
        self.data = data
        self.col_names = col_names
        pass

    def measure(self,input_base=None):
        sub_domains = [self.domains[at]+0.0 for at in self.att]
        bins = [np.arange(t+1) for t in sub_domains]

        if self.att == ():
            #TODO:how to handle two way marginals
            sparse_vec = np.array(len(self.data))
        else:
            datavector = np.histogramdd(self.data.values, bins)[0]
            datavector = datavector.flatten()
            # sparse_vec = csr_matrix(datavector)
            sparse_vec = datavector
        true_answer = mult_kron_vec(self.res_mat_list, sparse_vec)
        col_size = np.prod(sub_domains).astype(int)
        rd = np.sqrt(self.noise_level) * np.random.normal(size=[col_size, 1])
        cov_rd = mult_kron_vec(self.res_mat_list, rd)
        self.noisy_answer = true_answer + cov_rd
        #TODO: move it to class Mechnism
        self.true_answer = true_answer + np.zeros_like(cov_rd)
        #TODO: find a similar way as mult_kron_vec to add the true data
        self.trueData=sparse_vec.reshape(-1,1)


    def get_recon_answer(self, mat_list):
        self.recon_answer = mult_kron_vec(mat_list, self.noisy_answer)
        return self.recon_answer

    def get_origin_answer(self, mat_list):
        self.true_recon_answer = mult_kron_vec(mat_list, self.true_answer)
        return self.true_recon_answer
    
    def get_true_data(self):
        return self.trueData


class ResidualPlanner:

    def __init__(self, domains, bases=None):
        self.col_names = None
        self.data = None
        self.domains = domains
        self.bases = bases
        self.num_of_mech = 0
        self.num_of_res = 0

        self.marg_index = {}
        self.res_index = {}
        self.id2res = {}

        self.marg_dict = {}
        self.res_dict = {}

        self.pcost_coeff = {}
        self.var_coeff = {}
        self.var_bound = {}
        self.var_coeff_sum = defaultdict(int)

        self.sparse_row = {}
        self.sparse_col = {}

        self.pcost_sum = {}
        self.pcost_res = {}
        self.var_sum = {}
        self.var_res = {}

        self.residual_matrix = {}

    def preprocessing(self):
        pass

    def input_mech(self, att):
        pass

    def input_data(self, data, col_names):
        self.data = data
        self.col_names = col_names
        pass

    def output_coefficient(self):
        pass

    def get_noise_level(self):
        pass

    def measurement(self):
        print("Start Measurement, total number of cells: ", len(self.res_dict))
        for i, att in enumerate(self.res_dict.keys()):
            if i % 10_000 == 0 and i > 0:
                print("Measuring cell: ", i)
            res_mech = self.res_dict[att]
            cols = [self.col_names[idx] for idx in att]
            sub_data = self.data.loc[:, cols]
            input_base=[self.bases[idx] for idx in att]
            res_mech.input_data(sub_data, cols)
            res_mech.measure(input_base)

    def reconstruction(self):
        print("Start Reconstruction, total number of queries: ", len(self.marg_dict))
        #TODO: check where is the if Attt bhelong to A' part, then based on that part calculate phi 
        for i, att in enumerate(self.marg_dict.keys()):
            if i % 10_000 == 0 and i > 0:
                print("Reconstructing marginal: ", i)
            mech = self.marg_dict[att]
            #calculate closure A 
            att_subsets = all_subsets(att)
            noisy_answer = 0.0
            # todo: move it to class Mechanism
            true_answer = 0.0
            trueData=[]
            #for each A' in closure(A)
            for subset in att_subsets:
                res_mech = self.res_dict[subset]
                mat_list = []
                phi_list=[]

                for at in att: 
                    # if atti belong to A' part, then use the residual matrix to reconstruct as algo 4
                    if at in subset:
                        sub_mat = self.residual_matrix[self.domains[at]]
                        sub_pinv = np.linalg.pinv(sub_mat)
                        phi=sub_mat@sub_pinv
                        phi_list.append(phi)
                        mat_list.append(sub_pinv)
                    else:
                        # if atti belong to A part, then use the identity matrix to reconstruct as algo 4
                        one_mat = np.ones([self.domains[at], 1]) / self.domains[at]
                        mat_list.append(one_mat)
                        phi_list.append(one_mat) 

                recon_answer = res_mech.get_recon_answer(mat_list)
                noisy_answer += recon_answer
                #TODO: find a way to add W and put W for noisy and true answer.
                trueData.append(res_mech.get_true_data())
                recon_true = res_mech.get_origin_answer(mat_list)
                true_answer += recon_true
            mech.input_noisy_answer(noisy_answer)
            mech.input_true_answer(true_answer)
            mech.input_true_data(trueData)

    def get_mean_error(self, ord=1):
        error_list = []
        N = len(self.data)
        for att in self.marg_dict:
            mech = self.marg_dict[att]
            noisy_answer = mech.get_noisy_answer()
            true_answer = mech.get_true_answer()
            true_data=mech.get_true_data()
            #error2=np.linalg.norm(true_answer-true_data,ord=ord)
            l_error = np.linalg.norm(noisy_answer - true_answer, ord=ord)
            error_list.append(l_error / N)
            #error_list.append(error2 / N)
        mean_error = np.mean(error_list)
        return mean_error


class ResPlanSum(ResidualPlanner):

    def __init__(self, domains, bases=None):
        super().__init__(domains, bases)
        self.preprocessing()

    def preprocessing(self):
        for i, k in enumerate(self.domains):
            base = self.bases[i]
            Bs, R, Us, Ur = find_residual_basis_sum(k, base)
            self.residual_matrix[k] = R
            self.pcost_sum[i] = 1
            self.pcost_res[i] = np.max(np.diag(R.T @ R))
            self.var_sum[i] = np.trace(Us @ Us.T)
            self.var_res[i] = np.trace(Ur @ Ur.T)

    def input_mech(self, att):
        """Store coefficients for variance and privacy cost."""

        # att = (2, 3) means workload on attributes 2 and 3
        mech = MargMech(self.domains, att)
        self.marg_dict[att] = mech
        self.marg_index[att] = self.num_of_mech
        self.num_of_mech += 1

        # the subsets of (2, 3) are (,), (2,), (3,) and (2, 3)
        # these are residual basis needed to reconstruct workload on attribute (2, 3)
        att_subsets = all_subsets(att)
        for subset in att_subsets:
            if subset not in self.res_dict:
                pcost_res_list = [self.pcost_res[at] for at in subset]
                pcost_coeff = np.prod(pcost_res_list)
                # save the privacy cost coefficient for the residual mechanism
                self.pcost_coeff[subset] = pcost_coeff

                res_mech = ResMech(self.domains, subset, self.residual_matrix)
                self.res_dict[subset] = res_mech
                self.res_index[subset] = self.num_of_res
                self.id2res[self.num_of_res] = subset
                self.num_of_res += 1

        for subset in att_subsets:
            var_res_list = [self.var_res[at] for at in subset]
            # trace (A kron B) = trace(A) * trace(B)
            var_res_query = np.prod(var_res_list)
            var_sum_list = []
            for c in att:
                if c not in subset:
                    var_sum_list.append(self.var_sum[c])
            var_sum_query = np.prod(var_sum_list)
            self.var_coeff_sum[subset] += var_sum_query * var_res_query

            # check numeric overflow
            if var_sum_query * var_res_query < 0:
                print(subset, var_sum_query, var_res_query)
                print("numeric overflow! var<0: ", var_sum_query * var_res_query)

    def output_coefficient(self):
        pcost_coeff_list = []
        var_sum_list = []
        for res in self.res_index.keys():
            pcost_coeff_list.append(self.pcost_coeff[res])
            var_sum_list.append(self.var_coeff_sum[res])
        v = np.array(var_sum_list)
        p = np.array(pcost_coeff_list)
        return v, p

    def get_noise_level(self,zero=False):
        param_v, param_p = self.output_coefficient()
        x_sum, obj = find_var_sum_cauchy(param_v, param_p, c=1.0)
        if x_sum is None:
            print("failed to find a solution")
            return 0
        for i, noise_level in enumerate(x_sum):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            if zero:
                res_mech.input_noise_level(0)
            else:
                res_mech.input_noise_level(noise_level)
        return obj


class ResPlanMax(ResidualPlanner):

    def __init__(self, domains, bases=None):
        super().__init__(domains, bases)
        self.num_of_query = 0
        self.preprocessing()

    def preprocessing(self):
        for i, k in enumerate(self.domains):
            base = self.bases[i]
            Bs, R, Us, Ur = find_residual_basis_max(k, base)
            self.residual_matrix[k] = R
            self.pcost_sum[i] = 1
            self.pcost_res[i] = np.max(np.diag(R.T @ R))
            self.var_sum[i] = np.diag(Us @ Us.T)
            self.var_res[i] = np.diag(Ur @ Ur.T)

    def input_mech(self, att, var_bound=1.0):
        """Get variance for each query."""

        mech = MargMech(self.domains, att)
        self.marg_dict[att] = mech
        self.marg_index[att] = self.num_of_mech
        self.num_of_mech += 1

        # the subsets of (2, 3) are (,), (2,), (3,) and (2, 3)
        # these are residual basis needed to reconstruct workload on attribute (2, 3)
        att_subsets = all_subsets(att)
        for subset in att_subsets:
            if subset not in self.res_dict:
                pcost_res_list = [self.pcost_res[at] for at in subset]
                pcost_coeff = np.prod(pcost_res_list)
                # save the privacy cost coefficient for the residual mechanism
                self.pcost_coeff[subset] = pcost_coeff

                res_mech = ResMech(self.domains, subset, self.residual_matrix)
                self.res_dict[subset] = res_mech
                self.res_index[subset] = self.num_of_res
                self.id2res[self.num_of_res] = subset
                self.num_of_res += 1

        row_list = []
        col_list = []
        var_list = []
        # cur_id = self.marg_index[att]
        num_of_rows = np.prod([self.domains[c] for c in att])

        for subset in att_subsets:
            var_coeff_list = []
            for c in att:
                if c in subset:
                    var_coeff_list.append(self.var_res[c])
                if c not in subset:
                    var_coeff_list.append(self.var_sum[c])
            if len(var_coeff_list) == 0:
                var_coeff = 1.0
            else:
                var_coeff = reduce(lambda xx, yy: np.kron(xx, yy), var_coeff_list)

            # check the size
            if len(var_coeff_list) == 0:
                row_size = 1
            else:
                row_size = len(var_coeff)
            assert row_size == num_of_rows
            if row_size > 10**6:
                print(subset)
                print("number of rows too large!", row_size)

            if len(var_coeff_list) == 0:
                var_list.append(var_coeff)
            else:
                var_list.extend(var_coeff)
            row_ids = self.num_of_query + np.arange(row_size)
            row_list.extend(row_ids)
            sub_id = self.res_index[subset]
            col_list.extend([sub_id] * row_size)

        self.sparse_row[att] = np.array(row_list)
        self.sparse_col[att] = np.array(col_list)
        self.var_coeff[att] = np.array(var_list)
        self.var_bound[att] = np.array([var_bound] * int(num_of_rows))
        self.num_of_query += int(num_of_rows)

    def output_coeff(self):
        pcost_coeff_list = []
        var_coeff_list = []
        row_list = []
        col_list = []
        var_bound_list = []
        for res in self.res_index.keys():
            pcost_coeff_list.append(self.pcost_coeff[res])
        for mech in self.marg_index.keys():
            var_coeff_list.append(self.var_coeff[mech])
            row_list.append(self.sparse_row[mech])
            col_list.append(self.sparse_col[mech])
            var_bound_list.append(self.var_bound[mech])
        coeff = np.array(pcost_coeff_list)
        val = np.concatenate(var_coeff_list)
        row = np.concatenate(row_list)
        col = np.concatenate(col_list)
        A = sp.csr_matrix((val, (row, col)), shape=(
            self.num_of_query, self.num_of_res))
        b = np.concatenate(var_bound_list)
        return coeff, A, b

    def get_noise_level(self):
        coeff, A, b = self.output_coeff()
        # var, obj = find_var_max_cvxpy(coeff, A, b)
        var, obj = find_var_max_gurobi(coeff, A, b)
        for i, noise_level in enumerate(var):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            res_mech.input_noise_level(noise_level)
        return obj


def test_prefixsum():
    domains = [16]
    bases = ['P']
    system = ResPlanSum(domains, bases)
    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            cur_domains = [domains[c]+0.0 for c in subset]
            num_query = np.product(cur_domains)
            system.input_mech(subset)
            total += num_query
    print("total num of queries", total)
    return system, total


def test_prefixsum_max():
    domains = [16]
    bases = ['P']
    system = ResPlanMax(domains, bases)
    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            cur_domains = [domains[c]+0.0 for c in subset]
            num_query = np.product(cur_domains)
            system.input_mech(subset)
            total += num_query
    print("total num of queries", total)
    return system, total


def test_Adult():
    domains = [85, 9, 100, 16, 7, 15, 6, 5, 2, 100, 100, 99, 42, 2]
    col_names = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country', 'income>50K']
    bases = ['P', 'I', 'P', 'I', 'I', 'I', 'I', 'I', 'I', 'P', 'P', 'P', 'I', 'I']
    system = ResPlanSum(domains, bases)

    data = pd.read_csv("adult.csv")
    system.input_data(data, col_names)
    print("Len of adult dataset: ", len(data))

    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
    print("Total num of queries: ", total, "\n")
    return system, total


def test_simple_adult():
    # Define domains for education(3), marital(2), gender(2)
    domains = [3, 2, 2]
    col_names = ['education', 'marital', 'gender']
    # Using Identity basis for all columns since they are small domains
    bases = ['P', 'I', 'I']
    system = ResPlanSum(domains, bases)

    data = pd.read_csv("simple_adult.csv")
    system.input_data(data, col_names)
    print("Len of simple adult dataset: ", len(data))

    att = tuple(range(len(domains)))
    total = 0
    for i in range(2, 3):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
            break
    print("Total num of queries: ", total, "\n")
    return system, total


def test_age_income():
    # Define domains for age(85) and income(2)
    domains = [85, 2]
    col_names = ['age', 'income']
    # Using Prefix basis for age (large domain) and Identity basis for income (small domain)
    bases = ['P', 'P']
    system = ResPlanSum(domains, bases)

    data = pd.read_csv("age_income.csv")
    system.input_data(data, col_names)
    print("Len of age_income dataset: ", len(data))

    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
    print("Total num of queries: ", total, "\n")
    return system, total


def test_allkway_csv(n, d=5, k=3):
    domains = [n] * d
    col_names = [str(i) for i in range(d)]
    bases = ['P'] * d
    system = ResPlanMax(domains, bases)
    data_numpy = np.zeros([10_000, d])
    df = pd.DataFrame(data_numpy, columns=col_names)
    system.input_data(df, col_names)

    att = tuple(range(len(domains)))
    total = 0
    for i in range(0, k+1):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for t, subset in enumerate(subset_i):
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
            if t % 10_000 == 0 and t > 0:
                print("Selecting marginal: ", t)
    # print("total num of queries: ", total, "\n")
    return system, total



if __name__ == '__main__':
    start = time.time()

    pcost = 1

    #system, total = test_Adult()
    system, total = test_simple_adult()
    #system, total = test_age_income()  # Using the new age_income test
    obj = system.get_noise_level()
    print("obj: ", obj)
    system.measurement()
    system.reconstruction()
    l_error = system.get_mean_error(ord=1)
    print("Mean Error: ", l_error)

    end = time.time()
    print("time is: ", end-start)


