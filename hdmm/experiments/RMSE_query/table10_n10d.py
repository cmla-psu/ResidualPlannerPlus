from hdmm import workload, templates
import time
import numpy as np
from IPython import embed
import gc
#from svdb import svdb

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
   


def run_table3(attribute_Data,dims=[0,1,2,3]):
    written_dict={}
    for d in range(2,16):
        n = 10
        domain = [10] * d
        ns=tuple(domain)
        W = workload.DimKMarginals(ns, dims)
        temp = templates.Marginals(ns, True)
        svdb2= svdb(W)
        svdbout = np.sqrt(svdb2 / W.shape[0])
        print(svdbout)
        t0 = time.time()
        loss = temp.optimize(W)
        t1 = time.time()
        lossout = np.sqrt(loss / W.shape[0])
        #result_svdb=svdb(W)
        attribute_Data.add_data(d,t1-t0,lossout)
        print(t1-t0)


if __name__ == '__main__':
    fileName='table3.csv'
    attribute_Data = attributeData()
    with open(fileName, 'w') as f:
        f.write('d,avgtimes,vartimes,avgRMSE,varRMSE\n')
    output_dict={}
    for i in range(1):
        run_table3(attribute_Data,dims=[0,1,2,3])
        
    print("-----------------------------------------")
    print("d,avgtimes,vartimes,avgRMSE,varRMSE")
    for d in range(2, 16):
        avgtimes,vartimes=attribute_Data.get_time_for_n(d)
        avgRMSE,varRMSE=attribute_Data.get_RMSE_for_n(d)
        line = '%d, %.10f, %.10f,%.10f, %.10f' % (d, avgtimes,vartimes,avgRMSE,varRMSE)
        with open(fileName, 'a') as f:
            print(line)
            f.write(line + '\n')
    
    