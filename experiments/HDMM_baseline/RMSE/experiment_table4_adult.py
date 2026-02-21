from hdmm import workload, templates,matrix
import time
import numpy as np
from benchmarks import SmallKrons
from svdb import svdb_marg



def adult_marginal(dims):
    ns = (100, 100, 100, 99, 85, 42, 16, 15, 9, 7, 6, 5, 2, 2)
    W1 = workload.DimKMarginals(ns, dims)
    return W1


def adult_small():
    P = workload.Prefix
    I = workload.Identity
    M = workload.IdentityTotal

    # loan_amt, int_rate, annual_inc, installment, term, grade, sub_grade, home_ownership, state, settlement status, loan_status, purpose
    W1 = SmallKrons([P(100), P(100), P(100), P(99), P(85), I(42), I(16), I(15), I(9), I(7), I(6), I(5), I(2), I(2)],5000)
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

def run_table7(attribute_Data):

    for i in range(1,7):
        if i ==6:
            dims=[0,1,2,3]
        else: 
            dims=[i]
        W = adult_marginal(dims)
        ns = get_domain(W)
        temp = templates.Marginals(ns, True)

        t0 = time.time()
        loss = temp.optimize(W)
        t1 = time.time()
        lossout = np.sqrt(loss / W.shape[0])
        attribute_Data.add_data(i,t1-t0,lossout)
        line = '%d, %.6f, %.6f' % (i, t1-t0,lossout)
        print(line)

def run_table7_svdb():

    
    for i in range(1,7):
        if i ==6:
            dims=[0,1,2,3]
        else: 
            dims=[i]
        #dims.append(i)
        W = adult_marginal(dims)
        ns = get_domain(W)
        temp = templates.Marginals(ns, True)

        t0 = time.time()
        loss = temp.optimize(W)
        t1 = time.time()
        svdb2=   svdb_marg(W)
        svdbout = np.sqrt(svdb2 / W.shape[0])
        lossout = np.sqrt(loss / W.shape[0])
        #attribute_Data.add_data(i,svdbout,lossout)
        print("iway,svdb,loss")
        line = ' %d,%.10f, %.10f' % (i,svdbout,lossout)
        print(line)

#small marginal version 
def run_table7_small():

    
  
    W = adult_small()
    ns = get_domain(W)
    temp = templates.Marginals(ns, True)
    

    loss = temp.optimize(W)
    #svdb2=   svdb(W)
    #svdbout = np.sqrt(svdb2 / W.shape[0])
    lossout= np.sqrt(loss / W.shape[0])
    #attribute_Data.add_data(i,t1-t0,lossout)
    print("svdb,loss")
    line = ' %.10f, %.10f' % ( lossout,lossout)
    print(line)
if __name__ == '__main__':
    smallSVDB = True

    if smallSVDB == True:
        run_table7_small()
    
    
    else:
        fileName='table7.csv'
        attribute_Data = attributeData()
        with open(fileName, 'w') as f:
            f.write('n-way,avgtimes,vartimes,avgRMSE,varRMSE\n')
        for i in range(5):
            run_table7(attribute_Data)
            
        print("-----------------------------------------")
        print("n-way,avgtimes,vartimes,avgRMSE,varRMSE")
        for n in range(1, 7):
            avgtimes,vartimes=attribute_Data.get_time_for_n(n)
            avgRMSE,varRMSE=attribute_Data.get_RMSE_for_n(n)
            line = '%d, %.10f, %.10f,%.10f, %.10f' % (n, avgtimes,vartimes,avgRMSE,varRMSE)
            with open(fileName, 'a') as f:
                print(line)
                f.write(line + '\n')
        
    