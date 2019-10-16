'''
A DOcplex model of the QUBO formulation of the assignment problem
'''
import docplex.cp.model as cp
import docplex.cp.modeler as ct
import docplex.cp.parameters as params
import docplex.cp.solution as sol
import numpy as np
import math

class PermutationProblem:
    def __init__(self, num_locs, bigF, D, qty, alpha):
        self.num_locs = num_locs
        self.num_items = int(sum(qty))
        '''A note on F and D:
            F[i][j]: interaction frequency between items i and j
            F[0]: inherent popularity index
            F is on a per-group basis because each group has a unique combination on items
            D[i][j]: distance between location i and j
            D[0]: from depot
            D is so far shared among all groups.
        '''
        self.bigF = bigF.astype('int32')
        self.D = D.astype('int32')
        self.qty = qty.astype('int32')
        

        self.F = self.computeF(self.bigF, qty)
        #matrix Q is square matrix with side num_items*num_locs+1
        # Q is np.matrix
        Q = self.computeQ(self.F, self.D)
        P = self.computeP()
        perm = [[cp.binary_var() for j in range(self.num_locs)]for i in range(self.num_items)]
        m1 = np.matrix(perm).reshape((1,-1))
        m2 = np.matrix(perm).reshape((-1,1))

        dist = m1 * Q * m2 + alpha * np.dot(P,np.array(perm).flatten())

        ct1 = ct.all([ ct.count(perm[i],1)== 1 for i in range(self.num_items)])
        ct2 = ct.all([ ct.count(perm[:][j],1)==1 for j in range(self.num_locs)])
    
        self.model = cp.CpoModel()
        self.model.add(perm)
        self.model.add(ct1)
        self.model.add(ct2)
        self.model.add(ct.minimize(dist))
        
    def find_sku_index(self,i):
        '''returns the 1-based sku index for a given item'''
        # NOTE: item index is 0 based in qty
        _itemid = i+1
        for j in range(1,len(self.qty)):
            _itemid -= self.qty[j]
            if _itemid <= 0:
                return j
        raise ValueError("bad item index that exceeds sku limit")

    def computeF(self, bigF, qty):
        num_items = int(sum(qty))
        ret = np.zeros((num_items+1,num_items+1))
        
        for i in range(1, num_items+1):
            ret[i][0] = bigF[0][self.find_sku_index(i-1)]
            for j in range(i, num_items+1):
                ret[i][j] = bigF[self.find_sku_index(i-1)][self.find_sku_index(j-1)]
        ret = ret + ret.transpose()
        return ret       
    
    def computeP(self):
        '''The linear coefficients representing inherent popularity index and distance from depot'''
        P = np.zeros(shape=self.num_items*self.num_locs, dtype=int)
        for i in range(self.num_items):
            for j in range(self.num_locs):
                P[i*self.num_locs + j] = self.F[0][i] + self.D[0][j]
        return P
        
    def computeQ(self,F,D):
        '''compute the essential matrix Q'''
        F_no0 = F[1:,1:].copy()
        D_no0 = D[1:,1:].copy()
        
        F_0 = F[0, :].copy()
        D_0 = D[0, :].copy()
        
        # take f times d
        F_no0 = F_no0.reshape(-1,1)
        D_no0 = D_no0.reshape(1,-1)
        n = F.shape[0]-1
        m = D.shape[0]-1

        _Q = np.matmul(F_no0, D_no0)

        Q_bar=np.zeros((n*m+1, n*m+1))
        
        Q = np.zeros((n*m, n*m))
        for i in range(n):
            for j in range(n):
                a = i*n + j
                submatrix = _Q[a,:]
                submatrix = submatrix.reshape(m,m)
                upper_left_x, upper_left_y = (i*m, j*m)
                Q[upper_left_x:upper_left_x+m,upper_left_y:upper_left_y+m] += submatrix
        Q_bar[1:,1:] = Q

        F_0 = F_0[1:].reshape(-1,1)
        D_0 = D_0[1:].reshape(1,-1)
        q_0 = np.matmul(F_0,D_0)

        Q_bar[1:,0] = q_0.flatten()

        #return np.matrix(Q_bar)
        return np.matrix(Q)

    def solve(self):
        perm = []
        print("\nsolving model...\n")
        solution = self.model.solve(TimeLimit=2)
        print(type(solution))
        if solution:
            all_vars = solution.get_all_var_solutions()
            perm = [[all_vars[i*self.num_locs+j+1].get_value() for j in range(self.num_locs)] for i in range(self.num_items)]
            print(perm)
        else:
            print("No solution")
        
        return perm