import numpy as np


def diff_vec(n):
    size = n*(n-1)//2
    vec = np.zeros([n, size])
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            vec[i, k] = 1
            vec[j, k] = -1
            k += 1
    return vec


def subtract_matrix(k):
    """Return Subtraction matrix Sub_k."""
    mat = np.zeros([k-1, k])
    for i in range(k-1):
        mat[i, i] = 1
        mat[i, i+1] = -1
    return mat


def privacy_cost(mat):
    """Get the privacy cost given the strategy matrix."""
    diag = np.diag(mat.T @ mat)
    return np.max(diag)



def test_residual():
    """Test the case for 1 residual matrix."""
    n = 6
    B_sum = np.ones([1, n])
    R = subtract_matrix(n)
    S = R @ R.T
    S_inv = np.linalg.inv(S)
    p_mat = R.T @ S_inv @ R
    print("diagonal under unbounded dp", np.diag(p_mat + 1.0/n))

    B = np.concatenate([B_sum, R])
    diff_R = B @ diff_vec(n)
    # diff_R = np.array([[0, 0, 0, 0, 0, 0],
    #                    [2, 1, 1, -1, -1, 0],
    #                    [1, 2, 1, 1, 0, -1],
    #                    [1, 1, 2, 0, 1, 1]])
    cov = np.block([[np.ones([1, 1])*n, np.zeros([1, n - 1])],
                    [np.zeros([n - 1, 1]), S]])
    cov_inv = np.linalg.inv(cov)
    diff_p_mat = diff_R.T @ cov_inv @ diff_R
    print("diagonal under bounded dp", np.diag(diff_p_mat))


if __name__ == '__main__':
    n = 6
    B = np.random.randint(-10, 10, [n, n])
    diag_unbounded = np.diag(B.T @ B)
    B_bounded = B @ diff_vec(n)
    diag_bounded = np.diag(B_bounded.T @ B_bounded)
    print(diag_unbounded, np.max(diag_unbounded))
    print(diag_bounded, np.max(diag_bounded))
