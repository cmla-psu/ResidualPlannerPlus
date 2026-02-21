import numpy as np
import itertools
import scipy.sparse as sp
from collections import defaultdict
import cvxpy as cp
from .softmax import *
from .hdmm_convex import McKennaConvex
import gurobipy as gp
from gurobipy import GRB
from .parameter import options


def find_var_sum_cauchy(v, p, c=1.0):
    """Solve the optimization problem in appendix B.6.

    min sum(v * sigma^2)
    s.t. sum(p / sigma^2) == c
    """
    T = np.sum(np.sqrt(v * p))**2 / c
    x = np.sqrt(T * p / (c * v))
    return x, T


def find_var_max_cvxpy(coeff, A, b):
    """Solve the optimization problem.

    min sum(coeff / sigma^2)
    s.t. A sigma^2 <= b
    """
    size = len(coeff)
    x = cp.Variable(size)
    constraints = [x >= 0]
    constraints += [A @ x - b <= 0]
    obj = cp.Minimize(cp.sum(coeff @ cp.inv_pos(x)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    # print("obj:", obj.value)

    # scale the variable so that the privacy cost is 1
    return x.value * obj.value, obj.value

#
def find_var_max_gurobi(coeff, A, b, time_limit=10):
    """Solve the small optimization problem.

    min sum(1/x)
    s.t. A x <= b
    """
    env = gp.Env(params=options)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    #set time limit to inf
    #m.Params.TIME_LIMIT = time_limit
    m.setParam('NonConvex', 2)
    #m.setParam('TimeLimit', 5*60)
    # m.setParam(GRB.Param.OutputFlag, 0)
    size = np.shape(A)[1]
    x = m.addMVar(size, lb=1e-5)
    x_inv = m.addMVar(size, lb=1e-5)

    m.setObjective(coeff @ x_inv, sense=GRB.MINIMIZE)
    m.addConstr(A @ x <= b)
    for i in range(size):
        m.addConstr(x[i] * x_inv[i] == 1.0)
    m.optimize()
    obj = m.getObjective().getValue()
    return x.X * obj, obj


def subtract_matrix(k):
    """Return matrix C_k."""
    mat = np.zeros([k-1, k])
    for i in range(k-1):
        mat[i, i] = 1
        mat[i, i+1] = -1
    return mat


def subtract_matrix_v2(m):
    """
    Return an (m-1) x m matrix where the first column is all 1s, entries (i, i+1) are -1, and all other entries are 0.
    """
    mat = np.zeros((m-1, m))
    mat[:, 0] = 1
    for i in range(m-1):
        mat[i, i+1] = -1
    return mat


def all_subsets(att):
    """Return all subsets of a tuple."""
    length = len(att)
    subsets = [()]
    for i in range(1, length + 1):
        subset_i = list(itertools.combinations(att, i))
        subsets = subsets + subset_i
    return subsets


def sqrt_mat(X):
    """Calculate the decomposition X = B^T B."""
    vec, mat = np.linalg.eigh(X)
    idx = np.where(vec > 1e-5)[0]
    B = (mat[:, idx] * np.sqrt(vec[idx])).T
    return B


def haar_tree_residual(n):
    """
    Build the (n-1) x n residual part of the Haar binary tree.

    Row 1 (Bs = all-ones) is NOT included; only the "left − right" rows.

    For n=8 this produces:
      [ 1  1  1  1 -1 -1 -1 -1]   level 1: left half vs right half
      [ 1  1 -1 -1  0  0  0  0]   level 2
      [ 0  0  0  0  1  1 -1 -1]
      [ 1 -1  0  0  0  0  0  0]   level 3 (leaf pairs)
      [ 0  0  1 -1  0  0  0  0]
      [ 0  0  0  0  1 -1  0  0]
      [ 0  0  0  0  0  0  1 -1]

    Requires n to be a power of 2 and n >= 2.
    """
    assert n >= 2 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    rows = []
    block_size = n // 2
    while block_size >= 1:
        num_blocks = n // (2 * block_size)
        for b in range(num_blocks):
            row = np.zeros(n)
            start = b * 2 * block_size
            row[start:start + block_size] = 1
            row[start + block_size:start + 2 * block_size] = -1
            rows.append(row)
        block_size //= 2
    return np.array(rows)


def prefix_workload(k):
    #TODO: find the meaning of it 
    mat = np.zeros([k, k])
    for i in range(k):
        mat[i, :i+1] = 1
    return mat

def range_workload(k):
    """
    Returns a matrix of shape (k*(k+1)//2, k),
    where each row corresponds to a contiguous
    range [i, j] (0-based), and columns i..j
    are set to 1 in that row.
    """
    num_ranges = k * (k + 1) // 2   # number of contiguous subranges
    mat = np.zeros((num_ranges, k))
    
    row = 0
    for start in range(k):
        for end in range(start, k):
            mat[row, start:end+1] = 1
            row += 1
    return mat


def  find_residual_basis_sum(k, base):
    if base == 'P':
        work = prefix_workload(k)
    elif base == 'I':
        work = np.eye(k)
    elif base == 'R':
        work = range_workload(k)
    elif base == 'H':
        work = np.eye(k)
    else:
        work = None
    #Bs is the 1nT
    Bs = np.ones([1, k])

    # For Haar basis, use full Haar matrix as S_i and apply Algorithm 4
    # (construct subtraction matrix via Cholesky, same as McKennaConvex case)
    if base == 'H':
        Si = np.concatenate([Bs, haar_tree_residual(k)])  # full k x k Haar
        P1 = Si - Si @ Bs.T @ Bs / k
        Cov = P1.T @ P1
        regularization = 1e-10
        Cov += np.eye(Cov.shape[0]) * regularization
        L = np.linalg.cholesky(Cov)
        col_norms = np.linalg.norm(L, axis=0)
        num_cols_to_keep = k - 1
        keep_indices = np.argsort(col_norms)[-num_cols_to_keep:]
        keep_indices = np.sort(keep_indices)
        P2 = L[:, keep_indices]
        res_mat = P2.T
        l_mat = np.concatenate([Bs, res_mat]).T
        r_mat = work.T
        X = np.linalg.solve(l_mat, r_mat).T
        Us = X[:, 0].reshape(-1, 1)
        Ur = X[:, 1:]
        return Bs, res_mat, Us, Ur

    # strategy = ConvexDP(work)

    temp = McKennaConvex(k)
    temp.optimize(work)
    strategy = temp.strategy()

    #pmat_CA = strategy.T @ strategy

    #A is the strategy matrix Si
    A = strategy
    Si=A
    #Ua is 1nT @ pinv(Si )
    Ua = Bs @ np.linalg.pinv(A)

    #L = work @ np.linalg.pinv(A)
    # print("var:\n", np.trace(L @ L.T))

    # var_sum_query is Te from old algo 
    #var_sum_query = Ua @ Ua.T

    old_P=np.linalg.inv(A)-Bs.T @ Ua / k
    P1=Si-Si @ Bs.T @ Bs / k
    Cov=P1.T @ P1
    regularization=1e-10
    Cov+= np.eye(Cov.shape[0]) * regularization
    L = np.linalg.cholesky(Cov)
    # Calculate L2 norms of each column
    col_norms = np.linalg.norm(L, axis=0)
    # Number of columns to keep = k - (1+k-k) = k-1
    num_cols_to_keep = k-1
    # Get indices of columns with largest norms
    keep_indices = np.argsort(col_norms)[-num_cols_to_keep:]
    # Keep only columns with largest norms
    keep_indices = np.sort(keep_indices)  # Sort indices in ascending order
    P2 = L[:, keep_indices]
    subi2=P2.T

    sub_old=np.linalg.pinv(old_P)

    def check_orthogonality(Bs, sub_matrix, tol=1e-5):
        """
        Check if Bs @ pinv(sub_matrix) is close to zero matrix within tolerance.
        
        Args:
            Bs: The sum query matrix
            sub_matrix: Matrix to check orthogonality with
            tol: Tolerance for numerical precision, defaults to 1e-10
            
        Raises:
            ValueError if orthogonality check fails
        """
        pinv_sub = np.linalg.pinv(sub_matrix)
        check = Bs @ pinv_sub
        
        if not np.allclose(check, np.zeros_like(check), atol=tol):
            max_diff = np.max(np.abs(check))
            raise ValueError(
                f"Orthogonality check failed: Bs @ pinv(sub) should be close to zero matrix. "
                f"Max absolute value: {max_diff}"
            )
            
    check_orthogonality(Bs, subi2)

    #pmat_res = pmat_CA - 1.0 / var_sum_query
    #res_mat = sqrt_mat(pmat_res)
    res_mat=subi2
    l_mat = np.concatenate([Bs, res_mat]).T
    r_mat = work.T
    X = np.linalg.solve(l_mat, r_mat).T
    Us = X[:, 0].reshape(-1, 1)
    Ur = X[:, 1:]

    return Bs, res_mat, Us, Ur

#TODO: need to modify that 
def find_residual_basis_max(k, base):
    if base == 'P':
        work = prefix_workload(k)
    elif base == 'I':
        work = np.eye(k)
    elif base == 'R':
        work = range_workload(k)
    else:
        work = None
    Bs = np.ones([1, k])

    param_m, param_n = work.shape
    bound = np.ones(param_m)*1

    args = configuration()
    args.init_mat = 'id_index'

    args.maxitercg = 5
    args.theta = 1e-8
    args.sigma = 1e-8
    args.NNTOL = 1e-5
    args.TOL = 1e-5

    index = work
    basis = np.eye(param_n)

    mat_opt = matrix_query(args, basis, index, bound)
    mat_opt.optimize()
    mat_cov = mat_opt.cov/np.max(mat_opt.f_var)

    pmat_CA = np.linalg.inv(mat_cov)
    A = np.linalg.cholesky(pmat_CA).T

    # print("pcost:\n", np.max(pmat_CA))

    Ua = Bs @ np.linalg.pinv(A)
    var_sum_query = Ua @ Ua.T

    pmat_res = pmat_CA - 1.0 / var_sum_query
    res_mat = sqrt_mat(pmat_res)

    l_mat = np.concatenate([Bs, res_mat]).T
    r_mat = work.T
    X = np.linalg.solve(l_mat, r_mat).T
    Us = X[:, 0].reshape(-1, 1)
    Ur = X[:, 1:]

    return Bs, res_mat, Us, Ur


def mult_kron_vec(mat_ls, vec):
    """Fast Kronecker matrix vector multiplication."""
    V = vec.reshape(-1, 1)
    row = 1
    X = V.T
    for Q in mat_ls[::-1]:
        m, n = Q.shape
        row *= m
        X = Q.dot(X.reshape(-1, n).T)
    return X.reshape(row, -1)

def calculate_range_workload(datavector):
    """
    Given a 1D datavector (array-like), compute the sum of datavector[i..j]
    for all 0 <= i <= j < n, and return those sums in a 1D NumPy array.

    Returns
    -------
    range_sums : ndarray of shape (n*(n+1)//2,)
        Each entry corresponds to the sum of a contiguous subrange of datavector.
        The subranges are enumerated in row-major order:
           [0,0], [0,1], [0,2], ..., [1,1], [1,2], ..., [2,2], etc.
    """
    # 1. Compute prefix sums
    prefix = np.cumsum(datavector)
    n = len(datavector)

    # 2. Allocate space for all contiguous-range sums
    #    Number of subranges = n*(n+1)//2
    range_sums = np.zeros(n*(n+1)//2, dtype=datavector.dtype)

    # 3. Fill in the range sums using prefix sums
    idx = 0
    for start in range(n):
        for end in range(start, n):
            if start == 0:
                # sum(datavector[0..end]) = prefix[end]
                range_sums[idx] = prefix[end]
            else:
                # sum(datavector[start..end]) = prefix[end] - prefix[start-1]
                range_sums[idx] = prefix[end] - prefix[start - 1]
            idx += 1

    return range_sums
    


if __name__ == "__main__":
    # coeff = np.array([1, 0.571])
    # A = np.array([[0.123, 0.877], [0.354, 0.645], [1.0, 0]])
    # b = np.array([1, 1, 1])

    coeff = np.array([1, 1/3.0])
    A = np.array([[0.25, 0.75], [1.0, 0]])
    b = np.array([1, 1])
    x, obj = find_var_max_gurobi(coeff, A, b)
    print("obj: ", obj)
