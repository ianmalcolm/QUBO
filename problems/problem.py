import abc
import numpy as np
import math
import gc

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
        ret = np.zeros((size,size))
        for i in range(size):
            ret = AtA[i][i] - 2*bt_A[i]

        del AtA
        del bt_A
        print(ret.shape)
        return mtx.to_upper_triangular(ret)