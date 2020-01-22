import numpy as np
import itertools
import math

from problems.bunching import BunchingQAP
from problems.placement import PlacementQAP
from ports.dwave import Dwave
from ports.classical_simanneal import ClassicalNeal
from .exterior_penalty import ExteriorPenaltyMethod
import utils.index as idx

class OurHeuristic:
    def __init__(self,n,m,k,F,D, DIST_HOR, num_rows, num_cols):
        self.n = n
        self.m = m
        self.k = k
        self.F = F.copy()
        self.D = D.copy()
        self.DIST_HOR = DIST_HOR
        self.num_rows = num_rows
        self.num_cols = num_cols
    
    def run(self):
        #######bunching########
        bunch = BunchingQAP(
            self.m,
            self.n,
            self.k,
            self.F
        )

        solver = ClassicalNeal()

        bunch_method = ExteriorPenaltyMethod(bunch,solver,100)
        solution1 = bunch.solution_mtx((bunch_method.run())[0])
        print(solution1)

        #######bunch permutation/ aggregate QAP########
        g = np.zeros(shape=self.n)
        members = []
        for i in range(self.k):
            members.append([])
        for i in range(self.n):
            for j in range(self.k):
                if solution1[i][j]:
                    g[i]=j
                    members[j].append(i)
        
        bigF = np.zeros((self.k,self.k))
        for i1 in range(self.k):
            for i2 in range(i1 + 1,self.k):
                interaction = 0
                for item1,item2 in itertools.product(members[i1],members[i2]):
                    interaction += self.F[item1][item2]
                bigF[i1][i2] = interaction

        bigD = np.zeros((self.k,self.k))
        for j1 in range(self.k):
            for j2 in range(j1+1,self.k):
                bigD[j1][j2] = self.DIST_HOR * (j2-j1)

        aggregate_placement_problem = PlacementQAP(
            self.k,
            self.k,
            bigF,
            bigD
        )
        dwave_solver = Dwave()
        aggregate_method = ExteriorPenaltyMethod(
            aggregate_placement_problem,
            dwave_solver,
            100
        )
        solution2 = aggregate_placement_problem.solution_matrix(
            (aggregate_method.run())[0]
        )


        #######placement within bunches########
        ret = np.zeros((self.n, self.m))
        s = (int)(math.floor(self.m / self.k))
        bunch_size = (int)(math.ceil(self.n / self.k))

        locations = []
        for i in range(self.k):
            locations.append([])
        for i in range(self.k):
            for j in range(self.k):
                if solution2[i][j]:
                    for r in range(1,self.num_rows+1):
                        locations[i].append(idx.index_1_q_to_l_1(r,2*j,self.num_rows))
                        locations[i].append(idx.index_1_q_to_l_1(r,2*j+1,self.num_rows))

        solution3 = {}
        for i in range(self.k):
            bunch_i = np.sort(members[i])
            locations_i = np.sort(locations[i])

            bunch_i_idx_map = {}
            locations_i_idx_map = {}
            for a in range(bunch_size):
                bunch_i_idx_map[bunch_i[a]] = a
            for b in range(s):
                locations_i_idx_map[locations_i[b]] = b

            FPrime = np.zeros((bunch_size,bunch_size))
            DPrime = np.zeros((s,s))

            for i1,i2 in itertools.product(bunch_i,bunch_i):
                FPrime[bunch_i_idx_map[i1]][bunch_i_idx_map[i2]] = self.F[i1][i2]

            for j1,j2 in itertools.product(locations_i,locations_i):
                DPrime[locations_i_idx_map[j1]][locations_i_idx_map[j2]] = self.D[j1][j2]
            np.set_printoptions(threshold=np.inf)
            print(FPrime, DPrime)
            np.set_printoptions(threshold=6)
            fine_placement_problem = PlacementQAP(
                s,
                bunch_size,
                FPrime,
                DPrime,
                weight0=1000,
                alpha0=20,
                const_weight_inc=True
            )
            solver_i = Dwave()
            fine_placement_method = ExteriorPenaltyMethod(
                fine_placement_problem,
                solver_i,
                100
            )
            
            solution3[i] = fine_placement_problem.solution_matrix(
                (fine_placement_method.run())[0]
            )
            print(solution3[i])

            for r in range(bunch_size):
                for c in range(s):
                    ret[bunch_i[r]][locations_i[c]] = (solution3[i])[r][c]
        
        return ret
            
            


                

