from hdmm import workload, templates
import time
import numpy as np
from benchmarks_marginal import DimKKrons_kway_small,DimKKrons_small


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
   


def run_table_hdmm(attribute_Data,dims=[0,1,2,3]):
    for d in range(15,16):
        n = 10
        domain = [10] * d
        ns=tuple(domain)

        R = workload.AllRange  # Assuming this is already defined

        workloadList = [R(10) for _ in range(len(domain))]
        #W = workload.DimKMarginals(ns, dims)
        W= DimKKrons_kway_small(workloadList, 3)

        temp = templates.DefaultKron(ns, True)

        t0 = time.time()
        loss = temp.optimize(W)
        t1 = time.time()
        lossout = np.sqrt(loss / W.shape[0])

        # A = temp.strategy()
        # A.weights = A.weights.astype(np.float32)
        # A.dtype = np.float32
        # y = np.zeros(A.shape[0], dtype=np.float32)
        # t0 = time.time()
        # AtA1 = A.gram().pinv()
        # AtA1.weights = AtA1.weights.astype(np.float32)
        # AtA1.dtype = np.float32
        # At = A.T
        # At.dtype = np.float32
        # A1 = AtA1 @ At
        # A1.dot(y)
        # t1 = time.time()
        # lossout = np.sqrt(loss / W.shape[0])
        attribute_Data.add_data(d,t1-t0,lossout)
        print("-----------------------------------------")
        print("d,times of reconstruct ,loss")
        line = '%d, %.10f, %.10f' % (d, t1 - t0,lossout)
        print(line)


if __name__ == '__main__':
    fileName='tablehdmm_table8_journal_selection2.csv'
    attribute_Data = attributeData()
    with open(fileName, 'w') as f:
        f.write('d,avgtimes,vartimes,avgloss,varloss\n')
    for i in range(5):
        run_table_hdmm(attribute_Data,dims=[0,1,2,3])
        
    print("-----------------------------------------")
    print("d,avgtimes,vartimes,avgloss,varloss")
    for d in range(15,16):
        avgtimes,vartimes=attribute_Data.get_time_for_n(d)
        avgRMSE,varRMSE=attribute_Data.get_RMSE_for_n(d)
        line = '%d, %.10f, %.10f,%.10f, %.10f' % (d, avgtimes,vartimes,avgRMSE,varRMSE)
        with open(fileName, 'a') as f:
            print(line)
            f.write(line + '\n')
    
    