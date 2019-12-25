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
            k = num_groups
        
        decision matrix X is n by k
        decision vector is (n * k + #ancillary)
        F is n by n and item index is 0 based.
        '''
        self.m = num_locs
        self.n = num_items
        self.k = num_groups
        self.bunch_size = math.ceil(self.n / self.k)
        self.F = F.copy()
        self.q = self.generate_Q()

    @property
    def isExterior(self):
        return True
    
    @property
    def flow(self):
        return self.q['flow']

    @property
    def cts(self):
        ct1_m_0 = 10000.0
        ct1_alpha = 10
        ct2_m_0 = 10000.0
        ct2_alpha = 10
        cts = [(ct1_m_0, ct1_alpha, self.q['ct1']), (ct2_m_0, ct2_alpha, self.q['ct2'])]
        return cts

    def check(self, solution):
        '''
            solution is a dict of (var, val)
        '''
        #print(solution)
        solution_mtx = np.zeros((self.n, self.k), dtype=np.int8)
        for i in range(1,self.n+1):
            for k in range(1,self.k+1):
                index = idx.index_1_q_to_l_1(i,k,self.k) - 1
                solution_mtx[i-1][k-1] = solution[index]
        np.set_printoptions(threshold=np.inf)
        print(solution_mtx)
        np.set_printoptions(threshold=6)
        #recovers ancillaries. Not used for now.
        '''
        num_ancillaries = len(solution.keys()) - self.n*self.k
        num_bits = num_ancillaries / self.k
        twos = np.zeros(num_bits)
        for i in range(num_bits):
            twos[i] = math.pow(2,i)
        
        ancillaries = []
        start = self.n*self.k
        for i in range(self.k):
            ancillary_binary = np.zeros(num_bits, dtype=np.int8)
            for j in num_bits:
                offset = j + k * num_bits
                binary_index = start + offset
                ancillary_binary[j] = solution[binary_index]

            ancillary = np.dot(twos,ancillary_binary)
            ancillaries.append(ancillary)
        '''
        test_ct1 = True
        # check equality constraint ct1
        test = np.zeros(self.n, dtype=np.int8)
        for k in range(self.k):
            #sum(x_ik) = 1
            test += solution_mtx[:,k]
        result = test != 1
        if np.any(result):
            test_ct1 = False

        test_ct2 = True
        # check inequality constraint ct2
        test = np.zeros(self.k, dtype=np.int32)
        for i in range(self.n):
            test += solution_mtx[i,:]
        result = test > self.bunch_size
        if np.any(result):
            test_ct2 = False
        
        if test_ct1 and test_ct2:
            return [True, True]
        elif test_ct1 and not test_ct2:
            return [True, False]
        elif not test_ct1 and test_ct2:
            return [False, True]
        else:
            return [False, False]

    def generate_matrix_ct1(self):
        '''
        ct1: forall 1<=i<=n, sum(x_ik) = 1 forall 1<=k<=num_groups
        
        Remarks:
            A       (nm by nm) linear constraint matrix Ax = b
            b       column vector containing n 1's
            m1_0    initial penalty weight, for exterior method
            memmap is used for large matrices
        '''
        print("generating equality constraint")
        A = np.zeros(shape=(self.n*self.k, self.n*self.k),dtype=np.float32)
        for i in range(1,self.n+1):
            for k in range(1,self.k+1):
                x_ik_index = idx.index_1_q_to_l_1(i,k,self.k)
                # forall 1<=i<=n, (a)i,xik = 1 forall k, where 1<=k<=num_groups
                A[idx.index_1_to_0(i)][idx.index_1_to_0(x_ik_index)] = 1
        b = np.zeros((self.n * self.k),dtype=np.float32)
        for i in range(self.n):
            b[i] = 1
        bt_A = np.zeros(shape=self.n*self.k,dtype=np.float32)
        bt_A = np.matmul(b,A)
        # generalise linear terms to quadratic
        D = np.zeros(shape=(self.n*self.k, self.n*self.k),dtype=np.float32)
        for i in range(self.n*self.k):
            D[i][i] = bt_A[i]
        equality_mtx = np.zeros(shape=(self.n*self.k, self.n*self.k),dtype=np.float32)
        equality_mtx = np.matmul(np.transpose(A),A) - 2*D

        print(equality_mtx.shape)
        print("%d nonzeros out of %d" % (np.count_nonzero(equality_mtx), equality_mtx.shape[0]*equality_mtx.shape[1]))
        print("done")
        return equality_mtx

    def generate_flow_matrix(self):
        print("generating flow")
        ret = np.zeros(shape=(self.n*self.k,self.n*self.k),dtype=np.float32)
        for k in range(1,self.k+1):
            for i,j in itertools.product(range(1,self.n+1), range(1,self.n+1)):
                # NOTE: define interaction between identical items to be 0 because popularity is not relevant here.
                #if i==j:
                #    x_ik_idx_linear = idx.index_1_to_0(idx.index_1_q_to_l_1(i,k,self.k))
                #    ret[x_ik_idx_linear][x_ik_idx_linear] = -self.F[i-1][i-1]
                if i<j:
                    x_ik_idx_linear = idx.index_1_to_0(idx.index_1_q_to_l_1(i,k,self.k))
                    x_jk_idx_linear = idx.index_1_to_0(idx.index_1_q_to_l_1(j,k,self.k))
                    ret[x_ik_idx_linear][x_jk_idx_linear] = -self.F[i-1][j-1]
                else:
                    pass
        print("%d nonzeros out of %d" % (np.count_nonzero(ret), ret.shape[0]*ret.shape[1]))
        print("done")
        return ret
    
    def generate_matrix_ct2(self):
        print("generating inequality constraint")
        #compute number of binary ancillary vars
        num_bits = int.bit_length(self.bunch_size)
        num_ancillaries = num_bits * self.k
        
        # P(x,y) = x^t(A^tA-2D)x + b^tb + 2y^tAx - 2b^ty + y^ty
        # Each y_i = <2, u_i>
        ret_size = self.n*self.k + num_ancillaries
        ret = np.zeros((ret_size,ret_size))
        A = np.zeros((self.n*self.k, self.n*self.k))
        
        # forall 1<=k<=num_groups, sum(xik) <= s
        for k in range(1,self.k+1):
            for i in range(1,self.n+1):
                xik_idx_linear = idx.index_1_q_to_l_1(i,k,self.k)
                A[idx.index_1_to_0(k)][idx.index_1_to_0(xik_idx_linear)] = 1
        
        s = math.floor(self.m / self.k)
        b = np.zeros(self.n * self.k)
        for i in range(self.k):
            b[i] = s
        bt_A = np.matmul(np.transpose(b),A)
        print("A has %d nonzeros out of %d" % (np.count_nonzero(A), A.shape[0]*A.shape[1]))
        
        D = np.zeros((self.n*self.k, self.n*self.k))
        for i in range(self.n*self.k):
            D[i][i] = bt_A[i]

        print("D has %d nonzeros out of %d" % (np.count_nonzero(D), D.shape[0]*D.shape[1]))
        # A^tA-2D
        AtA_2D = np.matmul(np.transpose(A),A) - 2*D

        print("AtA-2D has %d nonzeros out of %d" % (np.count_nonzero(AtA_2D), AtA_2D.shape[0]*AtA_2D.shape[1]))
        ret[0:self.n*self.k, 0:self.n*self.k] = AtA_2D
        
        twos = np.zeros(num_bits)
        for i in range(num_bits):
            twos[i] = math.pow(2,i)

        # 2y^tAx
        for k in range(1, self.k+1):
            for i in range(1,self.n*self.k+1):
                for l in range(num_bits):
                    x_idx_1 = i
                    u_idx_1 = self.m*self.k + 1 + (k-1)*num_bits + l
                    # upper triangular only, x's index < u's index
                    ret[idx.index_1_to_0(x_idx_1)][idx.index_1_to_0(u_idx_1)] = 2*A[idx.index_1_to_0(k)][idx.index_1_to_0(i)] * twos[l]
        
        # -2b^ty, which is linear on y
        for k in range(1, self.k+1):
            for l in range(num_bits):
                u_idx_1 = self.m*self.k + 1 + (k-1)*num_bits + l
                ret[idx.index_1_to_0(u_idx_1)][idx.index_1_to_0(u_idx_1)] = (-2) * b[idx.index_1_to_0(k)] * twos[l]

        # y^ty, which is a quadratic form on the vector of y
        for k in range(1, self.k):
            for j,l in itertools.product(range(num_bits), range(num_bits)):
                uj_idx_1 = self.m*self.k + 1 + (k-1)*num_bits + j
                ul_idx_1 = self.m*self.k + 1 + (k-1)*num_bits + l
                if j==l:
                    ret[idx.index_1_to_0(uj_idx_1)][idx.index_1_to_0(uj_idx_1)] = twos[j]*twos[j]
                elif j<l:
                    ret[idx.index_1_to_0(uj_idx_1)][idx.index_1_to_0(ul_idx_1)] = 2*twos[j]*twos[l]
                else:
                    pass
        
        print(ret.shape)
        print("inequality mtx has %d nonzeros out of %d" % (np.count_nonzero(ret), ret.shape[0]*ret.shape[1]))
        test = np.transpose(ret) + ret
        print("test mtx has %d nonzeros out of %d" % (np.count_nonzero(ret), ret.shape[0]*ret.shape[1]))
        print("done")
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
        flow_matrix = self.generate_flow_matrix()
        print("flow matrix: ")
        print(flow_matrix)
        # process linear constraint
        equality_constraint_mtx = self.generate_matrix_ct1()
        print("equality matrix: ")
        print(equality_constraint_mtx)
        # process non-linear constraint
        inequality_constraint_mtx = self.generate_matrix_ct2()
        print("inequality matrix: ")
        print(inequality_constraint_mtx)
        ret['ct2'] = inequality_constraint_mtx

        #embed all matrices in big matrix with ancillaries
        _flow_matrix = np.zeros(inequality_constraint_mtx.shape)
        _flow_matrix[0:flow_matrix.shape[0], 0:flow_matrix.shape[0]] = flow_matrix

        ret['flow'] = _flow_matrix
        _equality_constraint_mtx = np.zeros(inequality_constraint_mtx.shape)
        _equality_constraint_mtx[0:equality_constraint_mtx.shape[0], 0:equality_constraint_mtx.shape[0]] = equality_constraint_mtx
        ret['ct1'] = _equality_constraint_mtx
        
        s = _flow_matrix + _equality_constraint_mtx + inequality_constraint_mtx
        s = np.transpose(s) +s
        print("sum mtx has %d nonzeros out of %d" % (np.count_nonzero(s), s.shape[0]*s.shape[1]))
        
        return ret