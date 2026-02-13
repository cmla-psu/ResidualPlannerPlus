from hdmm import workload, templates
import time
import numpy as np
from IPython import embed
import gc
import argparse
import numpy as np
import benchmarks
import pickle
import os
import time
from scipy.sparse.linalg import LinearOperator, eigs,svds
from hdmm.matrix import Identity
from scipy.sparse import coo_matrix
import math 
from calculate_variance import calculate_variance, calculate_variance_marginal,expected_error

def getratio(fileName,dims=[0,1,2,3]):

    for n in [2,3,4,5]:
        domain = [n,n,n,n,n]
        ns=tuple(domain)
       
        W = workload.DimKMarginals(ns, dims)
        #temp = templates.Marginals([n]*5)
        temp = templates.Marginals(ns, True)

        loss=temp.optimize(W)
        A = temp.strategy()
        #A.weights = A.weights.astype(np.float32)
        #A.dtype = np.float32
        #v1=calculate_variance(W,A)
  
        v =calculate_variance_marginal(W,A)
        v2=calculate_variance(W,A)
        lossout = np.sqrt(loss / W.shape[0])
        with open(fileName,'a') as f:
            line = '%d, %.6f, %.6f' % (n, v.min(),lossout)
            print(line)
            f.write(line+'\n')

if __name__ == '__main__':
    getratio("ratio_0424_2.csv")