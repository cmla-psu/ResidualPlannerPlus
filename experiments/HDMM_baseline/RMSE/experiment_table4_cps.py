from hdmm import workload, templates,matrix
import time
import numpy as np
from functools import reduce
from scipy import sparse
from benchmarks import SmallKrons
from svdb import svdb
def cps():
    #P = workload.Prefix
    M = workload.IdentityTotal
    V = workload.VStack

    W1 = workload.Kronecker([M(50), M(100), M(7), M(4), M(2)])
    #W2 = workload.Kronecker([P(50), P(100), M(7), M(4), M(2)])

    return V([W1])

def cps_marginal(dims):

    ns = (50,100,7,4,2)
    W1 = workload.DimKMarginals(ns, dims)

    return W1

def cps_small():
    P = workload.Prefix
    I = workload.Identity
    M = workload.IdentityTotal

    # loan_amt, int_rate, annual_inc, installment, term, grade, sub_grade, home_ownership, state, settlement status, loan_status, purpose
    W1 = SmallKrons([M(50), M(100), M(7), M(4), M(2)],5000)
    return W1

def get_domain(W):
    if isinstance(W, workload.VStack):
        W = W.matrices[0]
    if isinstance(W, workload.Weighted):
        W = W.base
    return tuple(Wi.shape[1] for Wi in W.matrices)

def svdb_marg(W):
    G = W.gram()
    d = len(G.domain)
    # create Y matrix
    Y = sparse.dok_matrix((2**d, 2**d))
    for a in range(2**d):
        for b in range(2**d):
            if b&a == a:
                Y[a,b] = G._mult[b]
    Y = Y.tocsr()
    
    # compute unique eigenvalues
    e = Y.dot(G.weights)
    # now compute multiplicities 
    mult = reduce(np.kron, [[1,n-1] for n in G.domain])
    
    ans = np.dot(mult, np.sqrt(e))**2 / mult.sum()
    
    return ans


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

def run_table6(attribute_Data):

    
    for i in range(1,7):
        if i ==6:
            dims=[0,1,2,3]
        else: 
            dims=[i]
        #dims.append(i)
        W = cps_marginal(dims)
        ns = get_domain(W)
        temp = templates.Marginals(ns, True)

        t0 = time.time()
        loss = temp.optimize(W)
        t1 = time.time()
        svdb2=   svdb_marg(W)
        svdbout = np.sqrt(svdb2 / W.shape[0])
        lossout = np.sqrt(loss / W.shape[0])
        attribute_Data.add_data(i,svdbout,lossout)


def run_table6_svdb():

    
    for i in range(1,7):
        if i ==6:
            dims=[0,1,2,3]
        else: 
            dims=[i]
        #dims.append(i)
        W = cps_marginal(dims)
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
def run_table6_small():

    
  
    W = cps_small()
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
        run_table6_svdb()
    
    
    else:
        fileName='table6.csv'
        attribute_Data = attributeData()
        with open(fileName, 'w') as f:
            f.write('n-ways,avgtimes,vartimes,avgRMSE,varRMSE\n')
        for i in range(5):
            run_table6(attribute_Data)
            
        print("-----------------------------------------")
        print("n-way,avgtimes,vartimes,avgRMSE,varRMSE")
        for n in range(1, 7):
            avgtimes, vartimes=attribute_Data.get_time_for_n(n)
            avgRMSE, varRMSE=attribute_Data.get_RMSE_for_n(n)
            line = '%d, %.10f, %.10f,%.10f, %.10f' % (n, avgtimes,vartimes,avgRMSE,varRMSE)
            with open(fileName, 'a') as f:
                print(line)
                f.write(line + '\n')
        
    