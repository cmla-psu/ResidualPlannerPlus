from hdmm import workload, templates
import time
import numpy as np
# from IPython import embed
import gc

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



def run_table1(n, d, timing_data,dims=[0,1,2,3]):
    written_dict={}
    # for n in [2,4,8,16,32,64,128,256,512,1024]:

    written_dict[n]={}
    domain = [n] * d
    ns=tuple(domain)
    W = workload.DimKMarginals(ns, dims)
    temp = templates.Marginals(ns, True)

    t0 = time.time()
    loss = temp.optimize(W)
    t1 = time.time()
    print("time: ", t1-t0)
    lossout = np.sqrt(loss / W.shape[0])

    timing_data.add_data(n,t1-t0,lossout)
    A = temp.strategy()
    return A


if __name__ == '__main__':
    fileName='table1.csv'
    attribute_Data = attributeData()
    # with open(fileName, 'w') as f:
    #     f.write('n,avgtimes,vartimes,avgRMSE,varRMSE\n')
    output_dict={}
    n = 3
    d = 15
    A = run_table1(n, d, attribute_Data, dims=[0, 1, 2, 3])

    cliques = []
    weights = []
    for wmat in A.matrices:
        weight = wmat.weight
        base = wmat.base
        attributes = []
        for i, mat in enumerate(base.matrices):
            if type(mat) == workload.Identity:
                attributes.append(str(i))
        cliques.append(tuple(attributes))
        weights.append(weight**2)
    print("num of marginals: ", len(cliques))
    dict = {}
    dict["cliques"] = cliques
    dict["weights"] = weights
    np.save("n" + str(n) + "d" + str(d) + ".npy", dict, allow_pickle=True)

