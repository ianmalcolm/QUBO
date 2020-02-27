from .problem import Problem
import itertools
import math
import numpy as np
import utils.index as idx
import os
import random

import dwavebinarycsp
import dimod
import operator

class BunchingQAP(Problem):
    def __init__(self, num_items, num_groups, F,
        euqality_weight=500,
        equality_alpha=10,
        inequality_weight=500,
        inequality_alpha=20):
        '''
        Let m = num_locs
            n = num_items
            k = num_groups
        
        decision matrix X is n by k
        decision vector is (n * k + #ancillary)
        F is symmetric, n by n and item index is 0 based.

        outputs:
            upper-triangular Q
        '''
        self.n = num_items
        self.k = num_groups
        self.bunch_size = math.ceil(self.n / self.k)
        self.F = F.copy()
        self.num_constraints = self.n + self.k
        self.num_ancillaries = 0
        self.ancillary_bit_length = 0

        self.euqality_weight = euqality_weight
        self.inequality_weight = inequality_weight
        self.equality_alpha = equality_alpha
        self.inequality_alpha = inequality_alpha

        #####mutable variables#####
        self.ms = []
        self.alphas = []
        self.canonical_A = -1
        self.canonical_b = -1
        self.count = 1
        #####end of state#####

        # construct initial Q      
        self.q = self.initialise_Q()


    @property
    def isExterior(self):
        return True
    
    @property
    def flow(self):
        return self.q['flow']

    @property
    def cts(self):
        cts = (self.ms, self.alphas, self.q['constraints'])
        return cts

    def initial(self):
        '''
        returns a (dict,energy) tuple of initial state that assigns each item to a random bunch.
        '''
        ret_dict = {}
        #randomly assign each item to some bunch
        for i in range(1,self.n+1):
            grp = random.randint(1,self.k)
            for k in range(1,self.k+1):
                index = idx.index_1_q_to_l_1(i,k,self.k) - 1
                if grp==k:
                    ret_dict[index] = 1
                else:
                    ret_dict[index] = 0
        
        for i in range(self.n*self.k, self.n*self.k + self.num_ancillaries):
            ret_dict[i] = random.randint(0,1)

        print("initial state dict has %d vars" % len(ret_dict.keys()))
        return (ret_dict,0)

    def solution_mtx(self, solution):
        '''
            param:
                solution: dict of (var,val)
            returns:
                np array of solution matrix

        '''
        solution_mtx = np.zeros((self.n, self.k), dtype=np.int8)
        for i in range(1,self.n+1):
            for k in range(1,self.k+1):
                index = idx.index_1_q_to_l_1(i,k,self.k) - 1
                solution_mtx[i-1][k-1] = solution[index]
        return solution_mtx        

    def check(self, solution):
        '''
            solution is a dict of (var, val)
        '''
        #print(solution)
        solution_mtx = self.solution_mtx(solution)
        
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
        
        return [test_ct1,test_ct2]

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

    def generate_matrix_ct1(self):
        '''
        ct1: forall 1<=i<=n, sum(x_ik) = 1 forall 1<=k<=num_groups
        
        Remarks:
            A       (n by nk) linear constraint coefficients
        '''
        print("generating equality constraint")
        A = np.zeros(shape=(self.n, self.n*self.k),dtype=np.float32)
        for i in range(1,self.n+1):
            for k in range(1,self.k+1):
                x_ik_index = idx.index_1_q_to_l_1(i,k,self.k)
                # forall 1<=i<=n, (a)i,xik = 1 forall k, where 1<=k<=num_groups
                A[idx.index_1_to_0(i)][idx.index_1_to_0(x_ik_index)] = 1
        b = np.ones(shape=self.n)
        weights = np.full(shape=self.n, fill_value=self.euqality_weight)
        #print(A)
        return A, b, weights
    
    def generate_matrix_ct2(self):
        print("generating inequality constraint")
        coeff = np.zeros(shape=(self.k,self.n*self.k + self.num_ancillaries))
        
        num_constraints = self.k
        
        twos = np.zeros(self.ancillary_bit_length)
        for i in range(self.ancillary_bit_length):
            twos[i] = math.pow(2,i)

        ancillary_startpos = self.n*self.k
        for k in range(1,num_constraints+1):
            # fill out the coefficients
            # forall 1<=k<=num_groups, sum(xik) <= s
            for i in range(1,self.n+1):
                xik_idx_linear = idx.index_1_q_to_l_1(i,k,self.k)
                coeff[idx.index_1_to_0(k)][idx.index_1_to_0(xik_idx_linear)] = 1
            # fill out the twos complement
            coeff[idx.index_1_to_0(k), ancillary_startpos: ancillary_startpos + self.ancillary_bit_length] = twos
            ancillary_startpos += self.ancillary_bit_length

        s = math.floor(self.n / self.k)
        b = np.full(shape=num_constraints, fill_value=s)
        weights = np.full(shape=num_constraints, fill_value=self.inequality_weight)

        return coeff, b, weights

    def generate_constraint_mtx(self):
        '''generate A for Ax=b'''
        #compute number of binary ancillary vars
        num_bits = int.bit_length(self.bunch_size)
        num_ancillaries = num_bits * self.k
        self.ancillary_bit_length = num_bits
        self.num_ancillaries += num_ancillaries
        
        size_A = self.n*self.k + self.num_ancillaries
        A = np.zeros(shape=(size_A,size_A))
        b = np.zeros(shape=size_A)
        weights = np.zeros(shape=self.n + self.k)

        ct1_coeff, ct1_b, ct1_weights = self.generate_matrix_ct1()
        ct1_len = ct1_coeff.shape[0]
        A[0:self.n, 0:(self.n*self.k)] = ct1_coeff
        b[0:ct1_len] = ct1_b
        weights[0:ct1_len] = ct1_weights
        
        ct2_coeff, ct2_b, ct2_weights = self.generate_matrix_ct2()
        
        ct2_len = ct2_coeff.shape[0]
        A[self.n: self.n+self.k, 0:size_A] = ct2_coeff
        b[ct1_len: (ct1_len+ct2_len)] = ct2_b
        weights[ct1_len: (ct1_len+ct2_len)] = ct2_weights
        #np.set_printoptions(threshold=np.inf)
        #print("ct2coeff: ", ct2_coeff)
        #np.set_printoptions(threshold=6)
        self.ms = weights[0:(ct1_len+ct2_len)]
        self.alphas = np.full(shape=(ct1_len+ct2_len),fill_value=10)
        self.alphas[0:ct1_len] = self.equality_alpha
        self.alphas[ct1_len:(ct1_len+ct2_len)] = self.inequality_alpha
        self.canonical_A = A.copy()
        self.canonical_b = b.copy()
        

        return super().A_to_Q(A, b, weights)
    
    def update_weights(self, solution):
        solution_arr = np.fromiter(solution.values(),dtype=np.int8)
        new_weights = np.zeros(self.num_constraints)
        for i in range(self.num_constraints):
            new_weights[i] = self.ms[i] + self.alphas[i]*abs(np.dot(self.canonical_A[i,:],solution_arr) - self.canonical_b[i])
        A = self.canonical_A.copy()
        b = self.canonical_b.copy()
        new_ct_mtx = super().A_to_Q(A,b,new_weights)
        
        #state udpate
        self.ms = new_weights
        self.q['constraints'] = new_ct_mtx
        return new_weights, new_ct_mtx

    def generate_dwavecsp(self):
        '''
            returns:
                a weighted constraint matrix generated by dwave's algorithm
        '''
        csp = dwavebinarycsp.ConstraintSatisfactionProblem('BINARY')

        def Aix_1(*args):
            return sum(list(args)) == 1
        for i in range(1,self.n+1):
            args = []
            for k in range(1,self.k+1):
                var_index = idx.index_1_q_to_l_1(i,k,self.k) - 1
                args.append(var_index)
            csp.add_constraint(Aix_1, args)
        
        def Aix_le_s(*args):
            return sum(list(args)) <= self.bunch_size
        for k in range(1,self.k+1):
            args = []
            for i in range(1,self.n+1):
                var_index = idx.index_1_q_to_l_1(i,k,self.k)-1
                args.append(var_index)
            print("adding %d inequality" % k)
            csp.add_constraint(Aix_le_s, args)

        print("stitching...")
        bqm = dwavebinarycsp.stitch(csp,max_graph_size=24)
        mtx = bqm.to_numpy_matrix()
        print(mtx)
        return 0
        
        

    def initialise_Q(self):
        '''
        returns: a dict containing two matrices.
            'flow': original negated flow terms to minimise
            'constraints': penalised constraint coefficients
        
        remarks:
            1. The returned values are combined with a sequence of penalty weights to get
            a sequence of QAP models to be sent for solving.
            
        '''
        ret = {}

        # process flow terms
        flow_matrix = self.generate_flow_matrix()
        #print("flow matrix: ")
        #print(flow_matrix)

        # process constraints
        constraint_mtx = self.generate_constraint_mtx()
        
        # embed flow into bigger matrix
        _flow_matrix = np.zeros(constraint_mtx.shape)
        _flow_matrix[0:flow_matrix.shape[0], 0:flow_matrix.shape[0]] = flow_matrix

        ret['flow'] = _flow_matrix
        ret['constraints'] = constraint_mtx
        return ret