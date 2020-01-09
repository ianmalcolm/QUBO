from .problem import Problem
import numpy as np
import utils.index as idx
import random

class PlacementQAP(Problem):
    def __init__(self, num_locs, num_items, F, D, gamma):
        '''
            F is n by n upper triangular with 0 based index
            D is m+1 by m+1, symmetric, with 1 based index
        '''
        
        self.m = num_locs
        self.n = num_items
        self.F = F.copy()
        self.F = np.transpose(self.F) + self.F
        for i in range(self.n):
            self.F[i][i] = self.F[i][i] / 2
        self.D = D.copy()
        self.num_constraints = m + n
        self.gamma = gamma

        self.ms = []
        self.alphas = []
        self.canonical_A = -1
        self.canonical_b = -1
        self.count = -1

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
        ret = {}
        for i in range(self.n):
            loc = random.randint(1,self.m)
            for j in range(1,self.m+1):
                index = idx.index_1_q_to_l_1(i,j,self.m) - 1
                if j==loc:
                    ret[index] = 1
                else:
                    ret[index] = 0
        return ret

    def check(self,solution):
        '''
            solution is a dict of (val, val)
        '''
        solution_mtx = np.zeros((self.n, self.m), dtype=np.int8)
        for i in range(1,self.n+1):
            for j in range(1,self.m+1):
                index = idx.index_1_q_to_l_1(i,j,self.m) - 1
                solution_mtx[i-1][j-1] = solution[index]
        np.set_printoptions(threshold=np.inf)
        print(solution_mtx)
        np.set_printoptions(threshold=6)
        
        test_ct1 = True
        test = np.zeros(self.n, dtype=np.int8)
        for i in range(self.m):
            test += solution_mtx[i,:]
        result = test !=1
        if np.any(result):
            test_ct1 = False
        
        test_ct2 = True
        test = np.zeros(self.m, dtype=np.int8)
        for i in range(self.m):
            test += solution_mtx[:,i]
        result = test != 1
        if np.any(result):
            test_ct2 = False
        
        return [test_ct1, test_ct2]

    def update_weights(self,solution):
        

    def initialise_flow_matrix(self):
        ret = np.zeros(self.m*self.n, self.m*self.n)
        for i in range(1,self.n+1):
            for j in range(1,self.n+1):
                for k in range(1,self.m+1):
                    for l in range(1,self.m+1):
                        # X is n by m
                        x_ik = idx.index_1_q_to_l_1(i,k,self.m)
                        x_jl = idx.index_1_q_to_l_1(j,l,self.m)
                        if x_ik == x_jl:
                            ret[x_ik-1][x_jl-1] = self.gamma * self.F[i-1][j-1] * self.D[0][k]
                        elif x_ik < x_jl:
                            ret[x_ik-1][x_jl-1] = self.F[i-1][j-1] * self.D[k][l]
        return ret

    def initialise_constraint_matrix(self):
        # prepare A
        A = np.zeros((self.m*self.n,self.m*self.n))
        for i in range(1,self.n+1):
            #ct1: each item in exactly one location
            #       forall i from 1 to n, sum(xik) = 1
            for k in range(1,self.m+1):
                x_ik = idx.index_1_q_to_l_1(i,k,self.m)
                A[i-1][x_ik] = 1

        for k in range(1, self.m+1):
            #ct2: each location has exactly one item
            #       forall k from 1 to m, sum(xik) = 1
            for i in range(1,self.n+1):
                x_ik = idx.index_1_q_to_l_1(i,k,self.m)
                A[k+self.n][x_ik] = 1
        
        # prepare b
        b = np.zeros(self.m*self.n)
        for i in range(self.num_constraints):
            b[i] = 1
        
        # prepare weights
        weights = np.full(shape=self.num_constraints, fill_value=10)
        
        self.canonical_A = A.copy()
        self.canonical_b = b.copy()
        self.ms = weights
        self.alphas = np.full(shape=self.num_constraints,fill_value=10)

        return super().A_to_Q(A,b,weights)

    def initialise_Q(self):
        '''
            minimise sum(i,j,k,l)(F_ij*D_kl*X_ik*X_jl)
        '''
        ret = {}
        flow_matrix = self.initialise_flow_matrix()
        constraint_matrix = self.initialise_constraint_matrix()
        
        ret['flow'] = flow_matrix
        ret['constraints'] = constraint_matrix
        return ret
        