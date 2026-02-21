import numpy as np
from hdmm.matrix import EkteloMatrix, VStack, Kronecker, Weighted
from hdmm import workload

import math



#This file is used to store all the function for calculating the max diag of variance 

def convert_implicit(A):
    if isinstance(A, EkteloMatrix) or isinstance(A, workload.ExplicitGram):
        return A
    return EkteloMatrix(A)

def expected_error(W, A, eps=np.sqrt(2), delta=0):
    """
    Given a strategy and a privacy budget, compute the expected squared error
    """
    assert delta == 0, 'delta must be 0'
    W, A = convert_implicit(W), convert_implicit(A)
    AtA = A.gram()
    AtA1 = AtA.pinv()
    WtW = W.gram()
    # appox proved by theorm 10 
    if isinstance(AtA1, workload.MarginalsGram):
        WtW = workload.MarginalsGram.approximate(WtW)
    X = WtW @ AtA1
    delta = A.sensitivity()
    if isinstance(X, workload.Sum):
        trace = sum(Y.trace() for Y in X.matrices)
    else:
        trace = X.trace()
    return trace/W.shape[0]

def calculate_variance(W, A, eps=np.sqrt(2), delta=0, normalize=False):
    """
    W @ AtA1 @ W.T == W@A1@A1.T@W.T 
    variance = diag(1/c(W@A1@A1.T@W.T))
    At current situation pcost is 1 
    """
    W, A = convert_implicit(W), convert_implicit(A)
    if isinstance(W, VStack):
        return np.concatenate([calculate_variance(Q, A, eps, delta, normalize) for Q in W.matrices])
    delta = A.sensitivity()
    var = 2.0/eps**2
    AtA1 = A.gram().pinv()
    X = W @ AtA1 @ W.T
    err = X.diag()
   
    return err

def calculate_variance_sampling(W, A, number=100000, eps=np.sqrt(2), normalize=False):
    """
    Computes the possible maxium diag of a workload W over opt solution as A,
    Under current codition we dont need eps and normalize.
    The number served as sampling draw if they are make in the situation of W is not 
    made of Marginals, then they will randomly selection rows from W's sub matrice and 
    generate a "marginal version of W" W' to get appox expected_error(W', A).
    For number=100000 times this process

    If W is marginal then according to line 178, all errors are the same. So by using the 
    expected_error, they get the trace of WtW @ AtA1 as max of diag of W.

    So for testing on our small dataset of smaller [n]*5 dataset, max(per_query_error2) result == min (per_query_error_sampling2)
    Therefore we are using this method to get appoxmation of larger dataset.
    
    Args:
        W (workload.Workload): the workload to evaluate.
        A (data.Data): opt solution of .
        number (int): the number of samples to draw (default: 100000). used in line 186
    
    Returns:
        The per-query error of the workload as an array of shape (number,).
    """
    # note: this only works for Kronecker or explicit strategy
    W, A = convert_implicit(W), convert_implicit(A)
    if isinstance(W, Weighted):
        ans = W.weight**2 * calculate_variance_sampling(W.base, A, number)
    #elif isinstance(W, VStack) and type(A) == VStack:
    #    m = W.shape[0]
    #    num = lambda Wi: int(number*Wi.shape[0]/m + 1)
    #    samples = [per_query_error_sampling(Wi,Ai.base,num(Wi)) for Wi,Ai in zip(W.matrices,A.matrices)]
    #    weights = [Ai.weight for Ai in A.matrices]
    #    ans = np.concatenate([err/w**2 for w, err in zip(weights, samples)])
    elif isinstance(W, VStack):
        m = W.shape[0]
        num = lambda Wi: int(number*Wi.shape[0]/m + 1)
        samples = [calculate_variance_sampling(Wi, A, num(Wi)) for Wi in W.matrices]
        ans = np.concatenate(samples)
    elif isinstance(W, Kronecker) and isinstance(A, Kronecker):
        assert isinstance(A, Kronecker)
        pieces=[calculate_variance_sampling(Wi, Ai, number) for Wi,Ai in zip(W.matrices,A.matrices)]
        ans = np.prod(pieces, axis=0)
    elif isinstance(W, Kronecker) and isinstance(A, workload.Marginals):
        # optimization: if W is Marginals, all errors are the same
        if all( type(Wi) in [workload.Identity, workload.Ones] for Wi in W.matrices ):
            err = expected_error(W, A)
            ans = np.repeat(err, number)
        else:
            # will be very slow, uses for loop
            AtA1 = A.gram().pinv()
            ans = np.zeros(number)
            for i in range(number):
                idx = [np.random.randint(Wi.shape[0]) for Wi in W.matrices]
                w = Kronecker([Wi[j] for Wi, j in zip(W.matrices, idx)])
                ans[i] = expected_error(w, A)
    else:
        ans = np.random.choice(calculate_variance(W, A), number)
        delta = A.sensitivity()
    #ans *= 2.0/eps**2
    return np.sqrt(ans) if normalize else ans


def calculate_variance_matrix(A, W):
    # using matrix.py's implementation to calculate_variance 
    #variance = diag(1/c(W@A1@A1.T@W.T))
    #using this function for verify in smaller dataset 
    
    A1 = A.pinv()
    product = W @ A1
    inverse_matrix = product.T
    product_result = product@inverse_matrix
    v = product_result.diag()
    tempreturn=0
    return tempreturn


def calculate_variance_numpy(A, W):
    # using pure numpy implementation to calculate_variance 
    #variance = diag(1/c(W@A1@A1.T@W.T))
    #using this function for verify in smaller dataset 

    A1 = np.linalg.pinv(A.dense_matrix())
    W=W.dense_matrix()
    product = np.matmul(W, A1)
    transpose_matrix = product.T
    v = np.matmul(product, transpose_matrix)
    v = np.diag(v).max()
    return v


def calculate_variance_marginal(W, A):
    """
    Computes the possible maxium diag of a workload W over opt solution as A,
    Under current codition we dont need eps and normalize.
    The number served as sampling draw if they are make in the situation of W is not 
    made of Marginals, then they will randomly selection rows from W's sub matrice and 
    generate a "marginal version of W" W' to get appox expected_error(W', A).
    For number=100000 times this process

    If W is marginal then according to line 178, all errors are the same. So by using the 
    expected_error, they get the trace of WtW @ AtA1 as max of diag of W.

    So for testing on our small dataset of smaller [n]*5 dataset, max(per_query_error2) result == min (per_query_error_sampling2)
    Therefore we are using this method to get appoxmation of larger dataset.
    
    Args:
        W (workload.Workload): the workload to evaluate.
        A (data.Data): opt solution of .
        number (int): the number of samples to draw (default: 100000). used in line 186
    
    Returns:
        The per-query error of the workload as an array of shape (number,).
    """
    # note: this only works for Kronecker or explicit strategy
    W, A = convert_implicit(W), convert_implicit(A)
    #Step 1 W==Vstack ( Marginal )
    if isinstance(W, VStack):
        samples = [calculate_variance_marginal(Wi, A) for Wi in W.matrices]
        ans = samples
    
    #Step 2 W' == W.matrices
    elif isinstance(W, Weighted):
        ans = W.weight**2 * calculate_variance_marginal(W.base, A)

    #Step 3 W''== W'.base and W'' is all constructed by Identity and Ones
    elif isinstance(W, Kronecker) and isinstance(A, workload.Marginals):
        # optimization: if W is Marginals, all errors are the same
        if all( type(Wi) in [workload.Identity, workload.Ones] for Wi in W.matrices ):
            err = expected_error(W, A)
            ans = err
        else:
            raise Exception("wrong Datatype for marginal W",W)
 
    else:
        raise Exception("wrong Datatype for marginal W",W)
        
    
    return ans