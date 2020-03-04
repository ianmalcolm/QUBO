import numpy as np

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
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.m):
                    for l in range(self.m):
                        energy += self.F[i][j] * self.D[k][l] * solution_mtx[i][k] * solution_mtx[j][l]
        return energy