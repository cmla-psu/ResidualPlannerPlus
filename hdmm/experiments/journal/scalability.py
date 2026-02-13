import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


from hdmm import workload, templates
import time
import numpy as np
#from IPython import embed
import gc



def fake_optimize(temp, W):
    temp._set_workload(W)
    params = temp._params
    for i in range(100):
        temp._loss_and_grad(params)

if True:
    with open('scalability_ls_0411.csv', 'w') as f:
        f.write('n,Kronecker,Marginals\n')

    for n in [2,4,8,16,32,64]:
        R = workload.AllRange(n)
        W = workload.Kronecker([R]*5)
        ns=tuple([n]*5)
        W1 = workload.DimKMarginals(ns, [0, 1, 2, 3])

        temp = templates.DefaultKron([n]*5)
        temp.optimize(W)
        A = temp.strategy()
        subs = []
        for sub in A.matrices:
            subs.append(workload.EkteloMatrix(sub.matrix.astype(np.float32)))
        A = workload.Kronecker(subs)
        y = np.zeros(A.shape[0], dtype=np.float32)
        t0 = time.time()
        A.pinv().dot(y)
        t1 = time.time()
        print('checkpt', n) 
        y = None
        gc.collect()

        #temp = templates.Marginals([n]*5)
        temp = templates.Marginals(ns, True)

        temp.optimize(W1)
        A = temp.strategy()
        A.weights = A.weights.astype(np.float32)
        A.dtype = np.float32
        y = np.zeros(A.shape[0], dtype=np.float32)
        t2 = time.time()
        AtA1 = A.gram().pinv()
        AtA1.weights = AtA1.weights.astype(np.float32)
        AtA1.dtype = np.float32
        At = A.T
        At.dtype = np.float32
        A1 = AtA1 @ At
        A1.dot(y)
        t3 = time.time()
        with open('scalability_ls_0410.csv','a') as f:
            line = '%d, %.6f, %.6f' % (n, t1-t0, t3-t2)
            print(line)
            f.write(line+'\n')

if False:
    with open('scalability_1d.csv', 'w') as f:
        f.write('n,Laplace,Gaussian\n')

    for k in range(1, 14):
        n = 2**k
        p = max(n//16, 1)
        W = workload.AllRange(n)
        temp1 = templates.McKennaConvex(n)
        temp2 = templates.PIdentity(p, n)
        t0 = time.time()
        #temp1.optimize(W, iters=100)
        fake_optimize(temp1, W)
        t1 = time.time()
        #temp2.optimize(W, iters=100)
        fake_optimize(temp2, W)
        t2 = time.time()
        dt1 = (t2 - t1)
        dt2 = (t1 - t0)
        with open('scalability_1d.csv', 'a') as f:
            line = '%d, %.6f, %.6f' % (n, dt1, dt2)
            print(line)
            f.write(line+'\n')

if False:
    with open('scalability_marg.csv', 'w') as f:
        f.write('d,Marginals')

    for d in range(2, 16):
        n = 10
        R = workload.AllRange(n)
        K = workload.Kronecker([R]*d)
        W = workload.VStack([K]*10)
        temp = templates.Marginals([n]*d, approx=False)
        t0 = time.time()
        fake_optimize(temp, W)
        t1 = time.time()
        with open('scalability_marg.csv', 'a') as f:
            print(d, t1-t0)
            f.write('%d, %.6f \n' % (d, t1-t0))

if False:
    with open('scalability_5d_0410.csv', 'w') as f:
        f.write('n,OPT_X,OPT_+,OPT_M,margin,lossMargin\n')

    for n in [2,4,8,16,32,64,128,256,512,1024]:
        domain = [n,n,n,n,n]
        ns=tuple(domain)
        #TODO change the work load to marginals like compute strategies with conresponding ns for three experiments in Figure3 
        R = workload.AllRange(n)
        K = workload.Kronecker([R,R,R,R,R])
        W = workload.VStack([K]*10)
        W1 = workload.DimKMarginals(ns, [0, 1, 2, 3])
        temp4 = templates.Marginals(ns, True)
        temp1 = templates.DefaultKron(domain, approx=False)
        temp2 = templates.DefaultUnionKron(domain, k=10, approx=False)
        temp3 = templates.Marginals(domain, approx=False)
        t0 = time.time()
        temp1.optimize(W)
        t1 = time.time()
        #temp2.optimize(W, iters=100)
        t2 = time.time()
        temp3.optimize(W)
        t3 = time.time()
        loss = temp4.optimize(W1,)
        t4 = time.time()
        lossout = np.sqrt(loss / W1.shape[0])
        with open('scalability_5d_0410.csv', 'a') as f:
            line = '%d, %.6f, %.6f, %.6f,%.6f,%.6f' % (n, t1-t0, t2-t1, t3-t2,t4-t3,lossout)
            print(line)
            f.write(line + '\n')

if False:
    with open('scalability_nd_0410.csv', 'w') as f:
        f.write('d,OPT_X,OPT_+,OPT_M,testMargin,lossMargin\n')

    for d in range(2, 16):
        n = 10
        domain = [10] * d
        ns = tuple(domain)
        R = workload.AllRange(n)
        K = workload.Kronecker([R]*10)
        W = workload.VStack([K]*10)
        W1 = workload.DimKMarginals(ns, [0, 1, 2, 3])
        temp1 = templates.DefaultKron(domain, approx=False)
        temp2 = templates.DefaultUnionKron(domain, k=10, approx=False)
        temp3 = templates.Marginals(domain, approx=False)
        temp4 = templates.Marginals(ns, True)
        t0 = time.time()
        temp1.optimize(W, iters=100)
        t1 = time.time()
        temp2.optimize(W, iters=100)
        t2 = time.time()
        temp3.optimize(W, iters=100)
        t3 = time.time()
        loss = temp4.optimize(W1, iters=100)

        t4 = time.time()
        lossout = np.sqrt(loss / W1.shape[0])
        with open('scalability_nd_0410.csv', 'a') as f:
            line = '%d, %.6f, %.6f, %.6f' % (n, t1-t0, t2-t1, t3-t2,t4-t3, lossout)
            print(line)
            f.write(line + '\n')

