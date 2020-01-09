import os
import math
import numpy as np

from orders.order_parser import OrderParser
from DistanceGenerator import DistanceGenerator
from methods.exterior_penalty import ExteriorPenaltyMethod
from problems.bunching import BunchingQAP
from problems.placement import PlacementQAP
from ports.dwave import Dwave
from ports.classical_simanneal import ClassicalNeal

import utils.mtx as mtx

NUM_SKUS = 10
WAREHOUSE_NUM_COLS = 6
WAREHOUSE_NUM_ROWS = 5
NUM_LOCS = WAREHOUSE_NUM_COLS*WAREHOUSE_NUM_ROWS
NUM_GROUPS = 3
DIST_VERTICAL = 1
DIST_HORIZONTAL = 1
ORDER_DIRNAME = 'orders'

group_num_cols = 2
group_num_rows = 5

def main():
    order_parser = OrderParser("orders/order.txt", NUM_SKUS, threshold=0)
    # F: (n by n) upper triangular interaction frequency matrix
    F = order_parser.gen_F()
    print(F)

    qty = order_parser.summary()

    problem = BunchingQAP(
        NUM_LOCS,
        NUM_LOCS,
        NUM_GROUPS,
        F
        )

    solver = ClassicalNeal()
    method = ExteriorPenaltyMethod(problem, solver)
    solution1 = method.run()

    ######################
    #second stage
    ######################
    
    def extract_F_Prime(F, index_list):
        '''
            index_set:      a sorted list of 0 based indices to be extracted
        '''
        size = len(index_list)
        FPrime = np.zeros((size,size))
        for i in range(size):
            for j in range(i,size):
                FPrime[i][j] = F[index_list[i]][index_list[j]]
        return FPrime

    #calculate D, m+1 by m+1
    D_gen = DistanceGenerator(
        WAREHOUSE_NUM_ROWS,
        WAREHOUSE_NUM_COLS, 
        DIST_VERTICAL, 
        DIST_HORIZONTAL,
        group_num_rows, 
        group_num_cols
        )
    D = D_gen.gen_S_shape()
    print("D: ", D)

    #extract Dprime, xy+1 by xy+1, distance matrix for the first few locations
    # NOTE: Usually Dprime cannot be extrapolated. Exception is no column crossing within a bunch
    Dprime = D_gen.gen_Dprime(D)
    print("Dprime: ", Dprime)

    for i in range(NUM_GROUPS):
        index_list = []
        items_choice_list = solution1[:,i]
        for j in range(items_choice_list):
            if items_choice_list[j]:
                index_list.append(j)
        Fprime = extract_F_Prime(F,index_list)
        problem2 = PlacementQAP(
            NUM_LOCS,
            NUM_LOCS,
            Fprime,
            Dprime,
            1
        )
        solver2 = Dwave()
        method2 = ExteriorPenaltyMethod(problem2,solver2)
        solution2 = method2.run()
        print(solution2)

if __name__ == "__main__":
    main()