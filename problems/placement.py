from .problem import Problem
import numpy as np
import utils.index as idx
import random

class PlacementQAP(Problem):
    def __init__(self, num_locs, num_items, F, D, gamma=1, weight0=10, alpha0=60, const_weight_inc=False,
        linear = None
        ):
        '''
            F is n by n symmetric with 0 based index
            D is m by m symmetric with 0 based index
            gamma is a factor multiplied to all linear terms
                gamma can be used to control the importance of
                absolute popularity versus interaction frequency.
            linear          a bunch of additional linear terms used during specialisation
        '''
        
        self.m = num_locs
        self.n = num_items
        self.F = F.copy()
        self.D = D.copy()
        self.num_constraints = self.m + self.n
        self.gamma = gamma
        self.weight0 = weight0
        self.alpha0 = alpha0
        self.CONST_WEIGHT_INC = const_weight_inc

        self.ms = []
        self.alphas = []
        self.canonical_A = -1
        self.canonical_b = -1
        self.count = -1

        self.linear = linear

        self.q = self.initialise_Q()
    @property
    def isExterior(self):
        return True
    
    @property
    def flow(self):
        print("flow sum: ", np.sum(np.sum(self.q['flow'],axis=0),axis=0))
        return self.q['flow']
    
    @property
    def cts(self):
        cts = (self.ms, self.alphas, self.q['constraints'])
        return cts
    
    def initial(self):
        ret = {}
        for i in range(1,self.n+1):
            loc = random.randint(1,self.m)
            for j in range(1,self.m+1):
                index = idx.index_1_q_to_l_1(i,j,self.m) - 1
                if j==loc:
                    ret[index] = 1
                else:
                    ret[index] = 0
        return (ret,0)

    @staticmethod
    def solution_matrix(solution,n,m):
        solution_mtx = np.zeros((n, m), dtype=np.int8)
        for i in range(1,n+1):
            for j in range(1,m+1):
                index = idx.index_1_q_to_l_1(i,j,m) - 1
                solution_mtx[i-1][j-1] = solution[index]
        return solution_mtx
    
    @staticmethod
    def check_mtx(solution_mtx):
        size = solution_mtx.shape[0]
        test_ct1 = True
        test = np.zeros(size)
        for i in range(size):
            test += solution_mtx[i,:]
        result = test !=1
        if np.any(result):
            test_ct1 = False
        
        test_ct2 = True
        test = np.zeros(size)
        for i in range(size):
            test += solution_mtx[:,i]
        result = test != 1
        if np.any(result):
            test_ct2 = False
        
        return [test_ct1, test_ct2]

    def check(self,solution):
        '''
            solution is a dict of (val, val)
        '''
        print(solution)
        solution_mtx = PlacementQAP.solution_matrix(solution, self.n, self.m)
        
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
        solution_arr = np.fromiter(solution.values(),dtype=np.int8)
        new_weights = np.zeros(self.num_constraints)
        for i in range(self.num_constraints):
            if self.CONST_WEIGHT_INC:
                new_weights[i] = self.ms[i] * self.alphas[i]
            else:
                new_weights[i] = self.ms[i] + self.alphas[i]*abs(np.dot(self.canonical_A[i,:],solution_arr) - self.canonical_b[i])
        A = self.canonical_A.copy()
        b = self.canonical_b.copy()
        new_ct_mtx = super().A_to_Q(A,b,new_weights)
        
        #state udpate
        self.ms = new_weights
        self.q['constraints'] = new_ct_mtx
        return new_weights, new_ct_mtx

    def initialise_flow_matrix(self):
        ret = np.zeros((self.m*self.n, self.m*self.n))
        for i in range(1,self.n+1):
            for j in range(1,self.n+1):
                for k in range(1,self.m+1):
                    for l in range(1,self.m+1):
                        # X is n by m
                        x_ik = idx.index_1_q_to_l_1(i,k,self.m)
                        x_jl = idx.index_1_q_to_l_1(j,l,self.m)
                        if x_ik == x_jl:
                            ret[x_ik-1][x_jl-1] = self.gamma * self.F[i-1][j-1] * self.D[k-1][k-1]
                        elif x_ik < x_jl:
                            ret[x_ik-1][x_jl-1] = self.F[i-1][j-1] * self.D[k-1][l-1]
        '''
        np.set_printoptions(threshold=np.inf)
        print("flow matrix: ", ret)
        np.set_printoptions(threshold=6)
        '''
        if not (self.linear is None):
            ret = ret + np.diag(self.linear)
        np.savetxt("flow.txt",ret,fmt='%d')
        return ret

    def initialise_constraint_matrix(self):
        # prepare A
        A = np.zeros((self.m*self.n,self.m*self.n))
        for i in range(1,self.n+1):
            #ct1: each item in exactly one location
            #       forall i from 1 to n, sum(xik) = 1
            for k in range(1,self.m+1):
                x_ik = idx.index_1_q_to_l_1(i,k,self.m)
                A[i-1][x_ik-1] = 1

        for k in range(1, self.m+1):
            #ct2: each location has exactly one item
            #       forall k from 1 to m, sum(xik) = 1
            for i in range(1,self.n+1):
                x_ik = idx.index_1_q_to_l_1(i,k,self.m)
                A[k+self.n - 1][x_ik-1] = 1
        '''
        np.set_printoptions(threshold=np.inf)
        print(A)
        np.set_printoptions(threshold=6)
        '''
        # prepare b
        b = np.zeros(self.m*self.n)
        for i in range(self.num_constraints):
            b[i] = 1
        
        # prepare weights
        weights = np.full(shape=self.num_constraints, fill_value=self.weight0)
        
        self.canonical_A = A.copy()
        self.canonical_b = b.copy()
        self.ms = weights
        self.alphas = np.full(shape=self.num_constraints,fill_value=self.alpha0)

        ret = super().A_to_Q(A,b,weights)
        np.savetxt("constraint.txt",ret,fmt='%d')
        return ret

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
        