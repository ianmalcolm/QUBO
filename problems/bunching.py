from .problem import Problem
import itertools
import math
import numpy as np
import utils.index as idx
import os


class BunchingQAP(Problem):
    def __init__(self, num_locs, num_items, num_groups, F):
        '''
        Let m = num_locs
            n = num_items
        
        decision matrix X is n by m
        decision vector is (n * m + #ancillary)
        '''
        self.m = num_locs
        self.n = num_items
        self.num_groups = num_groups
        self.bunch_size = math.ceil(self.n / self.num_groups)
        self.F = F
        
        ct1_m_0 = 0.1
        ct1_alpha = 10
        ct2_m_0 = 0.1
        ct2_alpha = 10
        q = self.generate_Q()
        self.flow = q['flow']
        cts = [(ct1_m_0, ct1_alpha, q['ct1']), (ct2_m_0, ct2_alpha, q['ct2'])]
        self.cts = cts

    @property
    def isExterior(self):
        return True
    
    @property
    def flow(self):
        return self.flow

    @property
    def cts(self):
        return self.cts

    def check(self, solution):
        return True

    def generate_matrix_ct1(self):
        '''
        ct1: forall 1<=i<=n, sum(x_ik) = 1 forall 1<=k<=num_groups
        
        Remarks:
            A       (nm by nm) linear constraint matrix Ax = b
            b       column vector containing n 1's
            m1_0    initial penalty weight, for exterior method
            memmap is used for large matrices
        '''
        A = np.memmap('data/A.dat',shape=(self.n*self.m, self.n*self.m),mode='w+',dtype=np.float32)
        for i in range(1,self.n+1):
            for k in range(1,self.num_groups+1):
                x_ik_index = idx.index_1_q_to_l_1(i,k,self.m)
                # forall 1<=i<=n, (a)i,xik = 1 forall k, where 1<=k<=num_groups
                A[idx.index_1_to_0(i)][idx.index_1_to_0(x_ik_index)] = 1
        b = np.zeros((self.n * self.m),dtype=np.float32)
        for i in range(self.n):
            b[i] = 1
        bt_A = np.zeros(self.n*self.m,dtype=np.float32)
        bt_A = np.matmul(b,A)
        print("check 2")
        D = np.zeros((self.n*self.m, self.n*self.m),dtype=np.float32)
        for i in range(self.n*self.m):
            D[i][i] = bt_A[i]
        print("check 3")
        return np.matmul(np.transpose(A),A) - 2*D

    def generate_flow_matrix(self):
        ret = np.memmap('data/flow.dat',shape=(self.n*self.m,self.n*self.m),mode='w+',dtype=np.float32)
        for k in range(1,self.num_groups+1):
            for i,j in itertools.product(range(1,self.n+1), range(1,self.n+1)):
                if i==j:
                    x_ik_idx_linear = idx.index_1_to_0(idx.index_1_q_to_l_1(i,k,self.m))
                    ret[x_ik_idx_linear][x_ik_idx_linear] = -self.F[i-1][i-1]
                elif i<j:
                    x_ik_idx_linear = idx.index_1_to_0(idx.index_1_q_to_l_1(i,k,self.m))
                    x_jk_idx_linear = idx.index_1_to_0(idx.index_1_q_to_l_1(j,k,self.m))
                    ret[x_ik_idx_linear][x_jk_idx_linear] = -self.F[i-1][j-1]
                else:
                    pass
        return ret
    
    def generate_matrix_ct2(self):
        #compute number of binary ancillary vars
        num_bits = int.bit_length(self.bunch_size)
        num_ancillaries = num_bits * self.num_groups
        
        # P(x,y) = x^t(A^tA-2D)x + b^tb + 2y^tAx - 2b^ty + y^ty
        # Each y_i = <2, u_i>
        ret_size = self.n*self.m + num_ancillaries
        ret = np.zeros((ret_size,ret_size))
        A = np.zeros((self.n*self.m, self.n*self.m))
        
        # forall 1<=k<=num_groups, sum(xik) <= s
        for k in range(1,self.num_groups+1):
            for i in range(1,self.n+1):
                xik_idx_linear = idx.index_1_q_to_l_1(i,k,self.m)
                A[idx.index_1_to_0(k)][idx.index_1_to_0(xik_idx_linear)] = 1
        
        s = math.floor(self.m / self.num_groups)
        b = np.zeros(self.n * self.m)
        for i in range(self.num_groups):
            b[i] = s
        bt_A = np.matmul(np.transpose(b),A)
        
        D = np.zeros((self.n*self.m, self.n*self.m))
        for i in range(self.n*self.m):
            D[i][i] = bt_A[i]

        ret[0:self.n*self.m, 0:self.n*self.m] = np.matmul(np.transpose(A),A) - 2*D
        
        twos = np.zeros(num_bits)
        for i in range(num_bits):
            twos[i] = math.pow(2,i)

        # all yi multiplies with x via A in + 2y^tAx
        for k in range(1, self.num_groups+1):
            for i in range(1,self.n*self.m+1):
                for l in range(num_bits):
                    x_idx_1 = i
                    u_idx_1 = self.m*self.n + 1 + (k-1)*num_bits + l
                    # upper triangular only, x's index < u's index
                    ret[idx.index_1_to_0(x_idx_1)][idx.index_1_to_0(u_idx_1)] = 2*A[idx.index_1_to_0(k)][idx.index_1_to_0(i)] * twos[l]
        
        # -2b^ty, which is linear on y
        for k in range(1, self.num_groups+1):
            for l in range(num_bits):
                u_idx_1 = self.m*self.n + 1 + (k-1)*num_bits + l
                ret[idx.index_1_to_0(u_idx_1)][idx.index_1_to_0(u_idx_1)] = (-2) * b[idx.index_1_to_0(k)] * twos[l]

        # y^ty, which is a quadratic form on the vector of y
        for k in range(1, self.num_groups):
            for j,l in itertools.product(range(num_bits), range(num_bits)):
                uj_idx_1 = self.m*self.n + 1 + (k-1)*num_bits + j
                ul_idx_1 = self.m*self.n + 1 + (k-1)*num_bits + l
                if j==l:
                    ret[idx.index_1_to_0(uj_idx_1)][idx.index_1_to_0(uj_idx_1)] = twos[j]*twos[j]
                elif j<l:
                    ret[idx.index_1_to_0(uj_idx_1)][idx.index_1_to_0(ul_idx_1)] = 2*twos[j]*twos[l]
                else:
                    pass

        return ret
    
    def generate_Q(self):
        '''
        returns: a dict containing three matrices.
            'flow': original negated flow terms to minimise
            'ct1': penalty matrix for ct1
            'ct2': penalty matrix for ct2
        
        remarks:
            1. The returned values are combined with a sequence of penalty weights to get
            a sequence of QAP models to be sent for solving.
            
        '''
        ret = {}

        # process flow terms
        print("generating flow")
        flow_matrix = self.generate_flow_matrix()
        print("Done generating flow")

        # process linear constraint
        print("generating equality constraint")
        equality_constraint_mtx = self.generate_matrix_ct1()
        print("done generating equality constraint")

        # process non-linear constraint
        print("generating inequality constraint")
        inequality_constraint_mtx = self.generate_matrix_ct2()
        print("done generating inequality constraint")
        ret['ct2'] = inequality_constraint_mtx

        #embed all matrices in big matrix with ancillaries
        _flow_matrix = np.zeros(inequality_constraint_mtx.shape)
        _flow_matrix[0:flow_matrix.shape[0], 0:flow_matrix.shape[0]] = flow_matrix
        ret['flow'] = _flow_matrix
        
        _equality_constraint_mtx = np.zeros(inequality_constraint_mtx.shape)
        _equality_constraint_mtx[0:equality_constraint_mtx.shape[0], 0:equality_constraint_mtx.shape[0]] = equality_constraint_mtx
        ret['ct1'] = _equality_constraint_mtx
        
        return ret