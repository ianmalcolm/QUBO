import numpy as np
import itertools

class QAPEvaluator:
    def __init__(self, n, m, F, D):
        '''
            F: n by n, symmetric
            D: m by m, symmetric
        '''
        self.F = F.copy()
        self.D = D.copy()
        self.n = n
        self.m = m

    def run(self, solution_mtx):
        '''
            solution_mtx: n by m
        '''
        energy = 0
        indices = []
        for i in range(self.n):
            for j in range(self.m):
                if solution_mtx[i][j]:
                    indices.append((i,j))
        
        for (i,k),(j,l) in itertools.product(indices, indices):
            energy += self.F[i][j]*self.D[k][l]
        
        return energy