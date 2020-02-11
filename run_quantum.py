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
from methods.ifhoos import IFHOOS
from ports.dwave import Dwave
from ports.classical_simanneal import ClassicalNeal
from sim.test_route import RouteEvaluator

import utils.mtx as mtx

RESULT_FOLDER = "simdata"

NUM_SKUS = 15
WAREHOUSE_NUM_COLS = 16
WAREHOUSE_NUM_ROWS = 4
NUM_LOCS = WAREHOUSE_NUM_COLS*WAREHOUSE_NUM_ROWS
NUM_GROUPS = 8
DIST_VERTICAL = 1
DIST_HORIZONTAL = 5
ORDER_DIRNAME = 'orders'

group_num_cols = 2
group_num_rows = 4
group_num_locs = group_num_cols * group_num_rows

NUM_ITERATIONS = 10

def main():
    for i in range(NUM_ITERATIONS):
        order_filename = 'order'+str(i)+'.txt'
        order_name = 'order'+str(i)
        order_path = os.path.join(ORDER_DIRNAME,order_filename)
        run(order_path, order_name)

def run(order_path, order_filename):
    order_parser = OrderParser(order_path, NUM_SKUS, threshold=0)
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

    ifhoos = IFHOOS(F,D)
    sol_ifhoos = ifhoos.run()
    print(sol_ifhoos)

    heuristic_qpu = OurHeuristic(
        NUM_LOCS,
        NUM_LOCS,
        NUM_GROUPS,
        F,
        D,
        DIST_HORIZONTAL,
        WAREHOUSE_NUM_ROWS,
        WAREHOUSE_NUM_COLS,
    )
    sol_heuristic_qpu = heuristic_qpu.run()
    t_heuristic_qpu = heuristic_qpu.get_timing()
    np.set_printoptions(threshold=np.inf)
    print("our heuristic has solution:\n", sol_heuristic_qpu)

    heuristic_sw = OurHeuristic(
        NUM_LOCS,
        NUM_LOCS,
        NUM_GROUPS,
        F,
        D,
        DIST_HORIZONTAL,
        WAREHOUSE_NUM_ROWS,
        WAREHOUSE_NUM_COLS,
        use_dwave=False
    )
    sol_heuristic_sw = heuristic_sw.run()
    t_heuristic_sw = heuristic_sw.get_timing()

    pureQAP = PureQAP(
        F,
        D
    )
    sol_pureQAP = pureQAP.run()
    t_pureQAP = pureQAP.get_timing()
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

    result_filename = str(NUM_SKUS) + "SKUs_" + str(WAREHOUSE_NUM_ROWS) + "*" + str(WAREHOUSE_NUM_COLS) + ".txt"
    with open(os.path.join(RESULT_FOLDER, result_filename), 'a+') as f:
        res_heuristic_qpu = evaluator.run(sol_heuristic_qpu)
        res_heuristic_sw = evaluator.run(sol_heuristic_sw)
        res_abc = evaluator.run(sol_abc)
        res_coi = evaluator.run(sol_coi)
        res_random = evaluator.run(sol_random)
        res_pure = evaluator.run(sol_pureQAP)
        res_ifhoos = evaluator.run(sol_ifhoos)

        res_list = list(map(str,[res_heuristic_qpu, res_heuristic_sw, res_abc, res_coi, res_random, res_pure, res_ifhoos]))
        res = " ".join(res_list) + '\n'
        f.write(res)

        str_heuristic_qpu = "result of our heuristic_qpu is " + str(res_heuristic_qpu) + '\n'
        print(str_heuristic_qpu)

        str_heuristic_sw = "result of our heuristic_sw is " + str(res_heuristic_sw) + '\n'
        print(str_heuristic_sw)
     

        str_abc = "result of abc is " + str(res_abc) + '\n'
        print(str_abc)

        str_coi = "result of coi is " + str(res_coi) + '\n'
        print(str_coi)

        str_random = "result of random is " + str(res_random) + '\n'
        print(str_random)

        str_pure = "result of pure QAP is " + str(res_pure) + '\n'
        print(str_pure)

        str_ifhoos = "result of pure QAP is " + str(res_ifhoos) + '\n'
        print(str_ifhoos)

    filename_t = str(NUM_SKUS) + "SKUs_" + str(WAREHOUSE_NUM_ROWS) + "*" + str(WAREHOUSE_NUM_COLS) + "_time.txt"
    with open(os.path.join(RESULT_FOLDER, filename_t), 'a+') as f:
        str_t = str(t_heuristic_qpu) + " " + str(t_heuristic_sw) + " " + str(t_pureQAP) + "\n"
        f.write(str_t)
    
def random_filename():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "res_"+current_time+".txt"

if __name__ == "__main__":
    main()