from hdmm import workload, templates

import numpy as np

from calculate_variance import calculate_variance, calculate_variance_marginal

def getratio(dims=[0,1,2,3]):

    for n in [2,3,4,5]:
        domain = [n,n,n,n,n]
        ns=tuple(domain)
       
        W = workload.DimKMarginals(ns, dims)
        temp = templates.Marginals(ns, True)

        loss=temp.optimize(W)
        A = temp.strategy()

  
        v =calculate_variance_marginal(W,A)
        v2=calculate_variance(W,A)
        print(sum(v)==loss)
        print(v)
        print(v2)
        lossout = np.sqrt(loss / W.shape[0])
        print(lossout)

        line = '%d, %.6f, %.6f' % (n, v.max(),v2.max())
        print(line)

if __name__ == '__main__':
    getratio()