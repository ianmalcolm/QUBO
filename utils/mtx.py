import numpy as np

def inspect_entries(F):
    size = F.shape[0]
    zeros = 0
    nonzeros = 0
    
    for i in range(size):
        for j in range(size):
            if F[i][j] == 0:
                zeros += 1
            else:
                nonzeros += 1
    
    return (zeros, nonzeros, zeros / nonzeros)

def inspect_upper(F):
    size = F.shape[0]
    zeros = 0
    nonzeros = 0
    
    for i in range(size):
        for j in range(size):
            if i<=j:
                if F[i][j] == 0:
                    zeros += 1
                else:
                    nonzeros += 1
    
    return (zeros, nonzeros, zeros / nonzeros)

def to_upper_triangular(matrix):
    size = matrix.shape[0]
    ret = matrix.copy()
    ret = ret + np.transpose(ret)
    for i in range(size):
        for j in range(size):
            if i==j:
                ret[i][i] = ret[i][i] / 2
            elif i>j:
                ret[i][j] = 0
    return ret