import os
import math
import numpy as np
from datetime import datetime

from orders.order_parser import OrderParser
from DistanceGenerator import DistanceGenerator
from methods.QAP import OurHeuristic
from methods.pureQAP import PureQAP
from methods.pureQAP_exact import ExactQAP
from methods.abc import ABCMethod
from methods.random import RandomMethod
from ports.dwave import Dwave
from ports.classical_simanneal import ClassicalNeal
from sim.test_route import RouteEvaluator

import utils.mtx as mtx

RESULT_FOLDER = "simdata"

NUM_SKUS = 15
WAREHOUSE_NUM_COLS = 8
WAREHOUSE_NUM_ROWS = 8
NUM_LOCS = WAREHOUSE_NUM_COLS*WAREHOUSE_NUM_ROWS
NUM_GROUPS = 8
DIST_VERTICAL = 1
DIST_HORIZONTAL = 5
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

    pureQAP = PureQAP(
        F,
        D
    )
    sol_pureQAP = pureQAP.run()
    print("pure QAP has solution:\n", sol_pureQAP)

    abc = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D), 3)
    sol_abc = abc.run()
    print("abc has solution:\n",sol_abc)

    coi = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D), NUM_LOCS)
    sol_coi = coi.run()

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

    filename = str(NUM_SKUS) + "SKUs_" + str(WAREHOUSE_NUM_ROWS) + "*" + str(WAREHOUSE_NUM_COLS) + ".txt"
    with open(os.path.join(RESULT_FOLDER, filename), 'a+') as f:
        res_heuristic = evaluator.run(sol_heuristic)
        res_abc = evaluator.run(sol_abc)
        res_coi = evaluator.run(sol_coi)
        res_random = evaluator.run(sol_random)
        res_pure = evaluator.run(sol_pureQAP)

        res = " ".join([res_heuristic, res_abc, res_coi, res_random, res_pure]) + '\n'
        f.write(res)

        str_heuristic = "result of our heuristic is " + str(res_heuristic) + '\n'
        print(str_heuristic)

        str_abc = "result of abc is " + str(res_abc) + '\n'
        print(str_abc)

        str_coi = "result of coi is " + str(res_coi) + '\n'
        print(str_coi)

        str_random = "result of random is " + str(res_random) + '\n'
        print(str_random)

        str_pure = "result of pure QAP is " + str(res_pure) + '\n'
        print(str_pure)

def random_filename():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "res_"+current_time+".txt"

if __name__ == "__main__":
    main()