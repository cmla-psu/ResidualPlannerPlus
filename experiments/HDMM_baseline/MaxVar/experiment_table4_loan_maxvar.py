from hdmm import workload, templates, matrix
import time
import numpy as np
from functools import reduce
import itertools
from calculate_variance import calculate_variance_marginal,calculate_variance_small_marginal


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def SmallKrons(blocks, size=5000):
    base = [workload.Total(W.shape[1]) for W in blocks]
    d = len(blocks)
    concat = []
    for attr in powerset(range(d)):
        subs = [blocks[i] if i in attr else base[i] for i in range(d)]
        tmp = reduce(lambda x,y: x*y, [blocks[i].shape[1] for i in attr], 1)
        W = workload.Kronecker(subs)
        if tmp <= size:
            concat.append(W)
    return workload.VStack(concat)

def loans_big():
    P = workload.Prefix
    I = workload.Identity
    # loan_amt, int_rate, annual_inc, installment, term, grade, sub_grade, home_ownership, state, settlement status, loan_status, purpose
    W1 = SmallKrons([I(101),I(101),I(101),I(101),I(3),I(8),I(36),I(6),I(51),I(4),I(5),I(15)],5000)
    return W1

def loans_marginal(dim):
    P = workload.Prefix
    I = workload.Identity
    ns= (101,101,101,101,3,8,36,6,51,4,5,15)
    # loan_amt, int_rate, annual_inc, installment, term, grade, sub_grade, home_ownership, state, settlement status, loan_status, purpose
    W1 = workload.DimKMarginals(ns, dim)
    return W1

def get_domain(W):
    if isinstance(W, workload.VStack):
        W = W.matrices[0]
    if isinstance(W, workload.Weighted):
        W = W.base
    return tuple(Wi.shape[1] for Wi in W.matrices)

def svdb(W):
    eigs = np.linalg.eig(W.gram().dense_matrix())[0]
    svdb = np.sqrt(np.maximum(0, np.real(eigs))).sum()**2 / W.shape[1]
    return svdb

class attributeData:
    def __init__(self):
        self.data = {}
    
    def add_data(self, n, time, loss_margin):
        if n not in self.data:
            self.data[n] = ([], [])
        self.data[n][0].append(time)
        self.data[n][1].append(loss_margin)
    
    def get_time_for_n(self, n):
        if n not in self.data:
            return None
        times = self.data[n][0]
        times=np.array(times)
        return times.mean(), times.var()
    
    def get_RMSE_for_n(self, n):
        if n not in self.data:
            return None
        RMSE = self.data[n][1]
        RMSE=np.array(RMSE)
        return RMSE.mean(), RMSE.var()

def run_table11(attribute_Data):

    for i in range(1,7):
        if i ==6:
            dims=[0,1,2,3]
        else: 
            dims=[i]
        W = loans_marginal(dims)
        ns = get_domain(W)
        temp = templates.Marginals(ns, True)

        t0 = time.time()
        loss = temp.optimize(W)
        t1 = time.time()
        lossout = np.sqrt(loss / W.shape[0])
        A = temp.strategy()
        v = calculate_variance_marginal(W, A)
        v = np.array(v)
        with open('table11.npy', 'wb') as f:
            np.save(f,v)
        attribute_Data.add_data(i, t1 - t0, v.max())
        line = '%d, %.10f, %.10f' % (i, t1 - t0,v.max())
        print(line)

def run_table11_small():
    W = loans_big()
    ns = get_domain(W)
    temp = templates.Marginals(ns, True)
    

    loss = temp.optimize(W)
    #svdb2=   svdb(W)
    #svdbout = np.sqrt(svdb2 / W.shape[0])
    lossout= np.sqrt(loss / W.shape[0])
    #attribute_Data.add_data(i,t1-t0,lossout)
    print("svdb,loss")
    A = temp.strategy()
    v=calculate_variance_small_marginal(W,A)
    v = np.array(v)
    print(v.max())


if __name__ == '__main__':
    smallMarginal = True

    if smallMarginal == True:
        run_table11_small()
    
    else:
        fileName='table11.csv'
        attribute_Data = attributeData()
        with open(fileName, 'w') as f:
            f.write('n-way,avgtimes,vartimes,avgMaxVar,varMaxVar\n')
        for i in range(5):
            run_table11(attribute_Data)
            
        print("-----------------------------------------")
        print("n-way,avgtimes,vartimes,avgMaxVar,varMaxVar")
        for n in range(1, 7):
            avgtimes,vartimes=attribute_Data.get_time_for_n(n)
            avgRMSE,varRMSE=attribute_Data.get_RMSE_for_n(n)
            line = '%d, %.10f, %.10f,%.10f, %.10f' % (n, avgtimes,vartimes,avgRMSE,varRMSE)
            with open(fileName, 'a') as f:
                print(line)
                f.write(line + '\n')
        
        