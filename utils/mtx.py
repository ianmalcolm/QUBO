import numpy as np
import itertools

def make_matrix(perm):
    n = len(perm)
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][perm[i]] = 1
    return matrix

def make_perm(mtx):
    n = mtx.shape[0]
    perm = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if mtx[i][j]:
                perm[i] = j
    return perm

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

def from_mtx_to_map(mtx):
    size = mtx.shape[0]
    ret = np.zeros(size)
    for j in range(size):
        for i in range(size):
            if mtx[i][j]:
                ret[j] = i
    return ret

def find_duplicate(arr):
    size = len(arr)
    d = {}
    for i in range(size):
        elem = arr[i]
        if not elem in d:
            d[elem] = 1
        else:
            d[elem] += 1
    
    for elem in d.keys():
        if d[elem] >= 2:
            return elem
    
    return None

def convert_to_int(arr):
    return [int(elem) for elem in arr]