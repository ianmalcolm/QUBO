import os
import math
import numpy as np

from orders.order_parser import OrderParser
from DistanceGenerator import DistanceGenerator
from methods.QAP import OurHeuristic
from methods.abc import ABCMethod
from methods.random import RandomMethod
from ports.dwave import Dwave
from ports.classical_simanneal import ClassicalNeal
from sim.test_route import RouteEvaluator

import utils.mtx as mtx

NUM_SKUS = 10
WAREHOUSE_NUM_COLS = 6
WAREHOUSE_NUM_ROWS = 4
NUM_LOCS = WAREHOUSE_NUM_COLS*WAREHOUSE_NUM_ROWS
NUM_GROUPS = 3
DIST_VERTICAL = 1
DIST_HORIZONTAL = 1
ORDER_DIRNAME = 'orders'

group_num_cols = 2
group_num_rows = 4
group_num_locs = group_num_cols * group_num_rows

def main():
    order_parser = OrderParser("orders/order.txt", NUM_SKUS, threshold=1)
    order_set = order_parser.gen_raw_orders()
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
    sol_heuristic = heuristic.run()
    np.set_printoptions(threshold=np.inf)
    print("our heuristic has solution:\n", sol_heuristic)
    
    abc = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D), 3)
    sol_abc = abc.run()
    print("abc has solution:\n",sol_abc)

    random = RandomMethod(NUM_LOCS, NUM_LOCS)
    sol_random = random.run()
    print("random has solution:\n",sol_random)

    evaluator = RouteEvaluator(
        qty,
        order_set,
        WAREHOUSE_NUM_COLS,
        WAREHOUSE_NUM_ROWS,
        DIST_VERTICAL,
        DIST_HORIZONTAL,
        NUM_LOCS,
        NUM_LOCS
        )
    res_heuristic = evaluator.run(sol_heuristic)
    print("result of our heuristic is %d" % res_heuristic)
    res_abc = evaluator.run(sol_abc)
    print("result of abc is %d" % res_abc)
    res_random = evaluator.run(sol_random)
    print("result of random is %d" % res_random)


if __name__ == "__main__":
    main()