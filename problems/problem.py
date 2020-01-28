import abc
import numpy as np
import math

import utils.mtx as mtx

class Problem(abc.ABC):
    '''
        An abstract problem contains a dict with:
        'flow'          the flow matrix
        'isExterior'    a boolean parameter specifying whether the problem is amenable to exterior penalty method.
                        if True, it has a non-zero 'alpha' and non-zero 'm_0's.
        'cts'           A constraint tuple. A constraint tuple is of the form (ms, alphas, mtx)
            
        every matrix should have the same dimension.

    '''
    @abc.abstractmethod
    def check(self, solution):
        pass

    @abc.abstractmethod
    def initial(self):
        pass

    @abc.abstractmethod
    def update_weights(self, new_weights):
        '''
            returns a tuple (weights, mtx)
                weights: updated penalty weights
                mtx: updated CONSTRAINT matrix
        '''
        pass
    
    @abc.abstractproperty
    def flow(self):
        pass

    @abc.abstractproperty
    def isExterior(self):
        pass
    
    @abc.abstractproperty
    def cts(self):
        pass

    def A_to_Q(self, A, b, penalty_weights):
        '''
        Converts the linear constraint Ax=b into quadratic coefficient matrix Q via square penalty
        inputs:
            A                   a numpy square matrix
            b                   a numpy vector
            penalty_weights     a list of positive floats
        
        '''
        size = A.shape[0]
        _root_penalty_weights = list(map(lambda x: math.sqrt(x), penalty_weights))
        num_constraints = len(penalty_weights)
        _A = A.copy()
        _b = b.copy()
        
        # populate penalty weights
        multiplicand_mtx = np.identity(size)
        for i in range(num_constraints):
            multiplicand_mtx[i][i] = _root_penalty_weights[i]

        _A = np.matmul(multiplicand_mtx,_A)
        _b = np.matmul(multiplicand_mtx,_b)

        # xt(AtA-2D)x, where D = diagonal generalisation of btA
        bt_A = np.matmul(_b,_A)
        D = np.zeros((size,size))
        for i in range(size):
            D[i][i] = bt_A[i]
        AtA = np.matmul(np.transpose(_A),_A)
        print("AtA has %d nonzeros out of %d" % (np.count_nonzero(AtA), AtA.shape[0]*AtA.shape[1]))
        ret = np.zeros((size,size))
        ret = AtA - 2*D
        return mtx.to_upper_triangular(ret)