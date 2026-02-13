from hdmm import workload, templates,matrix
import time
import itertools

import numpy as np
from benchmarks_marginal import SmallKrons

def DimKKrons(workloads, k=1):
    blocks = workloads
    base = [workload.Total(W.shape[1]) for W in blocks]
    d = len(blocks)
    
    concat = []
    for attr in itertools.combinations(range(d), k):
        subs = [blocks[i] if i in attr else base[i] for i in range(d)]
        W = workload.Kronecker(subs)
        concat.append(W)
    #return workload.VStack(concat) 
    return concat 


def adult_marginal(dims):
    ns = (100, 100, 100, 99, 85, 42, 16, 15, 9, 7, 6, 5, 2, 2)
    W1 = workload.DimKMarginals(ns, dims)
    return W1


def adult_big():
    R = workload.AllRange
    P = workload.Prefix
    M = workload.IdentityTotal
    I = workload.Identity
    T = workload.Total

    ns = (100,100,100,99,85,42,16,15,9,7,6,5,2,2)
    W_all = workload.DimKMarginals(ns, [0,1,2,3])
    line='W_marginal Shape is' + str(W_all.shape)
    print(line)
    W0 = DimKKrons([P(100), P(100), P(100), P(99), P(85), I(42), I(16), I(15), I(9), I(7), I(6), I(5), I(2), I(2)], 0)
    W1 = DimKKrons([P(100), P(100), P(100), P(99), P(85), I(42), I(16), I(15), I(9), I(7), I(6), I(5), I(2), I(2)], 1)
    W2 = DimKKrons([P(100), P(100), P(100), P(99), P(85), I(42), I(16), I(15), I(9), I(7), I(6), I(5), I(2), I(2)], 2)
    W3 = DimKKrons([P(100), P(100), P(100), P(99), P(85), I(42), I(16), I(15), I(9), I(7), I(6), I(5), I(2), I(2)], 3)
    #Adding all concat together 
    W_all_dim=W0+W1+W2+W3
    W_all=workload.VStack(W_all_dim)
    
    print('W_add Shape is' + str(W_all.shape))

    return W_all




def get_domain(W):
    if isinstance(W, workload.VStack):
        W = W.matrices[0]
    if isinstance(W, workload.Weighted):
        W = W.base
    return tuple(Wi.shape[1] for Wi in W.matrices)



if __name__ == '__main__':
    W_all=adult_big()
    ns = get_domain(W_all)



    temp1 = templates.DefaultKron(ns, True)
    temp2 = templates.DefaultUnionKron(ns, len(W_all.matrices), True)
    temp3 = templates.Marginals(ns, True)

    loss1 = temp1.optimize(W_all)
    loss2 = temp2.optimize(W_all)
    loss3 = temp3.optimize(W_all)

    losses = {}
    losses['kron'] = np.sqrt(loss1 / W_all.shape[0])
    losses['union'] = np.sqrt(loss2 / W_all.shape[0])
    losses['marg'] = np.sqrt(loss3 / W_all.shape[0])
   
    print(losses)
    


    

    
        
    