import os
import math
import numpy as np

from orders.order_parser import OrderParser
from DistanceGenerator import DistanceGenerator
from methods.QAP import OurHeuristic
from ports.dwave import Dwave
from ports.classical_simanneal import ClassicalNeal

import utils.mtx as mtx

NUM_SKUS = 10
WAREHOUSE_NUM_COLS = 6
WAREHOUSE_NUM_ROWS = 4
NUM_LOCS = WAREHOUSE_NUM_COLS*WAREHOUSE_NUM_ROWS
NUM_GROUPS = 3
DIST_VERTICAL = 1
DIST_HORIZONTAL = 5
ORDER_DIRNAME = 'orders'

group_num_cols = 2
group_num_rows = 4
group_num_locs = group_num_cols * group_num_rows

def main():
    order_parser = OrderParser("orders/order.txt", NUM_SKUS, threshold=1)
    # F: (n by n) upper triangular interaction frequency matrix
    F = order_parser.gen_F()
    print("F has shape: ", F.shape)
    np.set_printoptions(threshold=np.inf)
    print("F: ", F)
    np.set_printoptions(threshold=6)
    
    # D: (m by m) upper triangular distance matrix
    D_gen = DistanceGenerator(
        WAREHOUSE_NUM_ROWS,
        WAREHOUSE_NUM_COLS, 
        DIST_VERTICAL, 
        DIST_HORIZONTAL,
        group_num_rows, 
        group_num_cols
        )
    D = D_gen.gen_S_shape()
    np.set_printoptions(threshold=np.inf)
    print("D: ", D)
    np.set_printoptions(threshold=6)

    qty = order_parser.summary()

    heuristic = OurHeuristic(
        NUM_LOCS,
        NUM_LOCS,
        NUM_GROUPS,
        F,
        D,
        DIST_HORIZONTAL,
        WAREHOUSE_NUM_ROWS,
        WAREHOUSE_NUM_COLS
    )

    print("heuristic result: ", heuristic.run())
    input()

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

    #extract Dprime, xy+1 by xy+1, distance matrix for the first few locations
    # NOTE: Usually Dprime cannot be extrapolated. Exception is no column crossing within a bunch
    Dprime = D_gen.gen_Dprime(D)
    print("Dprime: ", Dprime)

    solution1_mtx = problem.solution_mtx(solution1[0])
    for i in range(NUM_GROUPS):
        index_list = []
        items_choice_list = solution1_mtx[:,i]

        for j in range(len(items_choice_list)):
            if items_choice_list[j]:
                index_list.append(j)
        Fprime = extract_F_Prime(F,index_list)
        np.set_printoptions(threshold=np.inf)
        print(Fprime)
        np.set_printoptions(threshold=6)
        problem2 = PlacementQAP(
            group_num_locs,
            group_num_locs,
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