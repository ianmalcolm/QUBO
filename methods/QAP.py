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
    def __init__(self,n,m,k,F,D, DIST_HOR, num_rows, num_cols, use_dwave=True):
        self.n = n
        self.m = m
        self.k = k
        self.F = F.copy()
        self.D = D.copy()
        self.DIST_HOR = DIST_HOR
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.use_dwave = use_dwave

        self.timing = 0
    
    def get_timing(self):
        return self.timing
    
    def run(self):
        #######bunching########
        bunch = BunchingQAP(
            self.n,
            self.k,
            self.F
        )

        solver = ClassicalNeal()

        bunch_method = ExteriorPenaltyMethod(bunch,solver,100)
        solution1 = bunch.solution_mtx((bunch_method.run())[0])

        self.timing += bunch_method.get_timing()

        #######grouping########
        group = BunchingQAP(
            self.m,
            self.k,
            -self.D
        )
        solver_group = ClassicalNeal()
        group_method = ExteriorPenaltyMethod(group,solver_group,100)
        # solution 1.5
        solution1_5 = group.solution_mtx((group_method.run())[0])

        self.timing += group_method.get_timing()

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

        locations = []
        for i in range(self.k):
            locations.append([])
        for i in range(self.m):
            for j in range(self.k):
                if solution1_5[i][j]:
                    locations[j].append(i)

        bigD = np.zeros((self.k,self.k))
        for j1 in range(self.k):
            for j2 in range(j1+1,self.k):
                distance = 0
                for loc1, loc2 in itertools.product(locations[j1], locations[j2]):
                    distance += self.D[loc1][loc2]
                bigD[j1][j2] = distance

        aggregate_placement_problem = PlacementQAP(
            self.k,
            self.k,
            bigF,
            bigD
        )
        dwave_solver = ClassicalNeal()
        aggregate_method = ExteriorPenaltyMethod(
            aggregate_placement_problem,
            dwave_solver,
            100
        )
        solution2 = aggregate_placement_problem.solution_matrix(
            (aggregate_method.run())[0],
            self.k,
            self.k
        )

        self.timing += aggregate_method.get_timing()

        #######placement within bunches########
        ret = np.zeros((self.n, self.m))
        s = (int)(math.floor(self.m / self.k))
        bunch_size = (int)(math.ceil(self.n / self.k))

        solution3 = {}
        np.set_printoptions(threshold=np.inf)
        print(members)
        print(locations)

        for i in range(self.k):
            bunch_i = np.sort(members[i])
            locations_i = np.sort(locations[i])

            # idx_map is from global to local
            # inverse is from local to global
            bunch_i_idx_map = {}
            bunch_i_idx_map_inv = {}
            locations_i_idx_map = {}
            locations_i_idx_map_inv = {}
            for a in range(bunch_size):
                bunch_i_idx_map[bunch_i[a]] = a
                bunch_i_idx_map_inv[a] = bunch_i[a]
            for b in range(s):
                locations_i_idx_map[locations_i[b]] = b
                locations_i_idx_map_inv[b] = locations_i[b]
            FPrime = np.zeros((bunch_size,bunch_size))
            DPrime = np.zeros((s,s))

            for i1,i2 in itertools.product(bunch_i,bunch_i):
                FPrime[bunch_i_idx_map[i1]][bunch_i_idx_map[i2]] = self.F[i1][i2]

            for j1,j2 in itertools.product(locations_i,locations_i):
                DPrime[locations_i_idx_map[j1]][locations_i_idx_map[j2]] = self.D[j1][j2]

            fine_placement_problem = PlacementQAP(
                bunch_size,
                s,
                FPrime,
                DPrime,
                weight0=100,
                alpha0=0.5,
                const_weight_inc=False
            )
            if self.use_dwave:
                solver_i = Dwave()
            else:
                solver_i = ClassicalNeal()
            fine_placement_method = ExteriorPenaltyMethod(
                fine_placement_problem,
                solver_i,
                100
            )
            
            solution3[i] = PlacementQAP.solution_matrix(
                (fine_placement_method.run())[0],
                bunch_size,
                s
            )
            self.timing += fine_placement_method.get_timing()
            
            np.set_printoptions(threshold=np.inf)
            print(bunch_i, locations_i)
            for local_item in range(bunch_size):
                for local_loc in range(s):
                    global_item = bunch_i_idx_map_inv[local_item]
                    global_loc = locations_i_idx_map_inv[local_loc]
                    if((solution3[i])[local_item][local_loc]):
                        print(global_item, global_loc)
                    ret[global_item][global_loc] = (solution3[i])[local_item][local_loc]
        return ret
            
            


                

