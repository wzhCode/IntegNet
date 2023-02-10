

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs


def scaled_laplacian(W):
    # Refer to https://github.com/VeritasYin/STGCN_IJCAI-18.git
    '''
    Normalized graph Laplacian function.
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    print(lambda_max)
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    # Refer to https://github.com/VeritasYin/STGCN_IJCAI-18.git
    '''
    Chebyshev polynomials approximation function.
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def weight_matrix(file_path):
    '''
    Weighted adjacency matrix load function.
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input weight matrix file was not found in {file_path}.')
    return W
