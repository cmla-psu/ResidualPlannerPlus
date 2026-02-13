from hdmm import workload, templates
import time
import numpy as np
# from IPython import embed
import gc
import itertools



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
        return times.mean(),times.var()

    def get_RMSE_for_n(self, n):
        if n not in self.data:
            return None
        RMSE = self.data[n][1]
        RMSE=np.array(RMSE)
        return RMSE.mean(),RMSE.var()


def nd_big(n):
    R = workload.AllRange
    P = workload.Prefix
    M = workload.IdentityTotal
    I = workload.Identity
    T = workload.Total

  
    W0 = DimKKrons([P(n), P(n),I(n), I(n), I(n)], 0)
    W1 =  DimKKrons([P(n), P(n),I(n), I(n), I(n)], 1)
    W2 =  DimKKrons([P(n), P(n),I(n), I(n), I(n)], 2)
    W3 =  DimKKrons([P(n), P(n),I(n), I(n), I(n)], 3)
    #Adding all concat together 
    W_all_dim=W0+W1+W2+W3
    W_all=workload.VStack(W_all_dim)
    
    print('nd W_add Shape is' + str(W_all.shape))

    return W_all


def run_table10():
    written_dict={}
    # for n in [2,4,8,16,32,64,128,256,512,1024]:
    for n in [2,4,8,16,32,64,128,256,512,1024]:
        written_dict[n]={}
        domain = [n] * 5
        ns=tuple(domain)
        W_all = workload.DimKMarginals(ns, [0,1,2,3])
        line='CPS W_marginal Shape is' + str(W_all.shape)
        print(line)

        W_all = nd_big(n)

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

        print("-----------------------------------------")
        print("n,times of reconstruct ,loss")
        line = '%d, %.10f, %.10f' % (n, t1 - t0,losses)
        print(losses)


if __name__ == '__main__':
    run_table1(n, d, attribute_Data, dims=[0, 1, 2, 3])

   

