import numpy as np
from problems.problem import Problem
import math
import gc
import utils.mtx as mtx

A = np.random.rand(3600*40,3600*40)
b = np.ones(shape=3600*40,dtype=np.int32)
penalty_weights = np.full(shape=3600*40, fill_value=10, dtype=np.int32)

def A_to_Q(A, b, penalty_weights):
    '''
    Converts the linear constraint Ax=b into quadratic coefficient matrix Q via square penalty
    inputs:
        A                   a numpy square matrix
        b                   a numpy vector
        penalty_weights     a list of positive floats
    
    '''
    gc.collect()
    size = A.shape[0]
    num_constraints = len(penalty_weights)
    print("copying")
    _A = A.copy()
    _b = b.copy()
    print("done.")

    print("populate penalty on A")
    for i in range(num_constraints):
        _A[i,:] = math.sqrt(penalty_weights[i]) * _A[i,:]
    print("done.")

    print('populate penalty on b')
    for i in range(num_constraints):
        _b[i] = math.sqrt(penalty_weights[i]) * _b[i]
    print("done")

    # xt(AtA-2D)x, where D = diagonal generalisation of btA
    print("computing btA")
    bt_A = np.dot(_b,_A)
    print("done")

    print("compute AtA")
    AtA = np.dot(np.transpose(_A),_A)
    print("done")

    del _A
    del _b
    #print("AtA has %d nonzeros out of %d" % (np.count_nonzero(AtA), AtA.shape[0]*AtA.shape[1]))
    for i in range(size):
        AtA[i][i] = AtA[i][i] - 2*bt_A[i]
    
    del bt_A
    ret = AtA
    print(ret.shape)
    return mtx.to_upper_triangular(ret)
    # xt(AtA-2D)x, where D = diagonal generalisation of btA
    print("computing btA")
    bt_A = np.dot(_b,_A)
    print("done")

    print("compute AtA")
    AtA = np.dot(np.transpose(_A),_A)
    print("done")

    del _A
    del _b
    #print("AtA has %d nonzeros out of %d" % (np.count_nonzero(AtA), AtA.shape[0]*AtA.shape[1]))
    for i in range(size):
        AtA[i][i] = AtA[i][i] - 2*bt_A[i]
    
    del bt_A
    ret = AtA
    print(ret.shape)
    return mtx.to_upper_triangular(ret)

print(A_to_Q(A,b,penalty_weights))