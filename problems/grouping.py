'''
A DOcplex model of the grouping problem
'''
import docplex.cp.model as cp
import docplex.cp.modeler as ct
import docplex.cp.parameters as params
import docplex.cp.solution as sol
import numpy as np
import math

class GroupingProblem:
    def __init__(self, k, F, time_limit=2):
        self.timing = 0
        self.F = F.astype('int32')
        self.n = F.shape[0]
        self.k = k
        self.group_size = self.n / self.k
        self.time_limit =time_limit
        
        self.model = cp.CpoModel()
        dvar_grouping = np.array([[cp.binary_var() for j in range(self.k)] for i in range(self.n)])
        self.model.add(dvar_grouping)

        ct1 = ct.all([ct.count(dvar_grouping[x], 1) == 1 
            for x in range(self.n)])
        ct2 = ct.all([ct.count(dvar_grouping[:,s], 1) <= self.group_size 
            for s in range(self.k)])

        self.model.add(ct1)
        self.model.add(ct2)

        flow = ct.sum([ 
            self.F[i][j]*dvar_grouping[i][s]*dvar_grouping[j][s]
            for s in range(self.k)
            for i in range(self.n) 
            for j in range(i, self.n) 
            ])
        
        self.model.add(ct.maximize(flow))
    
    def get_timing(self):
        return self.timing

    def solve(self):
        grouping = []
        print("\nsolving model...\n")
        solution = self.model.solve(TimeLimit=self.time_limit)
        self.timing = solution.get_solve_time()
        if solution:
            all_vars = solution.get_all_var_solutions()
            grouping = [[all_vars[i*self.k+j].get_value() for j in range(self.k)] for i in range(self.n)]
            print(grouping)
        else:
            print("No solution")
        return grouping
    
    def solution_mtx(self, solution):
        return np.array(solution, dtype=np.int32)