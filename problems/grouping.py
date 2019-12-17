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
    def __init__(self, group_size, num_skus, F):
        self.F = F.astype('int32')
        self.group_size = group_size
        self.num_skus = num_skus
        self.num_items = int(sum(self.F[0]))
        self.num_groups = math.ceil(self.num_items / self.group_size)
 
        a=0
        item_to_sku = np.zeros((self.num_items,)).astype('int32')
        for i in range(1,num_skus+1):
            for j in range(self.F[0][i]):
                item_to_sku[a] = i
                a += 1
        

        self.model = cp.CpoModel()
        print(self.num_groups, self.num_items)
        dvar_grouping = np.array([[cp.binary_var() for j in range(self.num_groups)] for i in range(self.num_items)])
        self.model.add(dvar_grouping)

        ct1 = ct.all([ct.count(dvar_grouping[x], 1) == 1 
            for x in range(self.num_items)])
        ct2 = ct.all([ct.count(dvar_grouping[:,s], 1) <= self.group_size 
            for s in range(self.num_groups)])

        self.model.add(ct1)
        self.model.add(ct2)

        flow = ct.sum([ 
            self.F[item_to_sku[i]][item_to_sku[j]]*dvar_grouping[i][s]*dvar_grouping[j][s]
            for s in range(self.num_groups)
            for i in range(self.num_items) 
            for j in range(i, self.num_items) 
            ])
        
        self.model.add(ct.maximize(flow))
    
    def solve(self):
        grouping = []
        print("\nsolving model...\n")
        solution = self.model.solve(TimeLimit=600)
        print(type(solution))
        if solution:
            all_vars = solution.get_all_var_solutions()
            grouping = [[all_vars[i*self.num_groups+j].get_value() for j in range(self.num_groups)] for i in range(self.num_items)]
            print(grouping)
        else:
            print("No solution")
        
        return grouping