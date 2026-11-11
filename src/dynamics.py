import numpy as np

def construct_H(M):
    K = M * (M - 1) // 2 # number of independent params in antisymmetric matrix
    H = np.zeros((K, M, M)) # K template matrices of size M x M

    a = 0
    for i in range(M):
        for j in range(i + 1, M): # just upper triangular part of A
            H[a, i, j] = 1
            H[a, j, i] = -1
            a += 1

    return H