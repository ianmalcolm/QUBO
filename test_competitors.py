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

NUM_SKUS = 5
WAREHOUSE_NUM_COLS = 16
WAREHOUSE_NUM_ROWS = 4
NUM_LOCS = WAREHOUSE_NUM_COLS*WAREHOUSE_NUM_ROWS
NUM_GROUPS = 8
DIST_VERTICAL = 1
DIST_HORIZONTAL = 3
ORDER_DIRNAME = 'orders'

group_num_cols = 2
group_num_rows = 4
group_num_locs = group_num_cols * group_num_rows

NUM_ITERATIONS = 10

def main():
    test_abc = []
    test_coi = []
    test_ifhoos = []
    for i in range(NUM_ITERATIONS):
        order_filename = 'order'+str(i)+'.txt'
        order_name = 'order'+str(i)
        order_path = os.path.join(ORDER_DIRNAME,order_filename)
        str_abc, str_coi, str_ifhoos = run(order_path, order_name)
        test_abc.append(str_abc)
        test_coi.append(str_coi)
        test_ifhoos.append(str_ifhoos)
    print(test_abc)
    print(test_coi)
    print(test_ifhoos)

def run(order_path, order_filename):
    order_parser = OrderParser(order_path, NUM_SKUS, threshold=0)
    F = order_parser.gen_F()
    qty = order_parser.summary()

    D_gen = DistanceGenerator(
        WAREHOUSE_NUM_ROWS,
        WAREHOUSE_NUM_COLS, 
        DIST_VERTICAL, 
        DIST_HORIZONTAL,
        group_num_rows,
        group_num_cols
        )
    D = D_gen.gen_S_shape()
    D_euclidean = D_gen.gen_Euclidean()

    order_parser = OrderParser(order_path, NUM_SKUS, threshold=0)
    order_set = order_parser.gen_raw_orders()

    ifhoos = IFHOOS(F,D)
    sol_ifhoos = ifhoos.run()

    abc = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D_euclidean), 8)
    sol_abc = abc.run()

    coi = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D), NUM_LOCS)
    sol_coi = coi.run()

    print(qty)
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
    res_abc = evaluator.run(sol_abc)
    res_coi = evaluator.run(sol_coi)
    res_ifhoos = evaluator.run(sol_ifhoos)
    
    print(mtx.from_mtx_to_map(sol_abc))
    print(mtx.from_mtx_to_map(sol_coi))
    print(mtx.from_mtx_to_map(sol_ifhoos))
    
    str_abc = "result of abc is " + str(res_abc) + '\n'
    print(str_abc)

    str_coi = "result of coi is " + str(res_coi) + '\n'
    print(str_coi)
    
    str_ifhoos = "result of ifhoos is " + str(res_ifhoos) + '\n'
    print(str_ifhoos)

    print(np.diag(D))
    print(np.diag(D_euclidean))

    return res_abc, res_coi, res_ifhoos

if __name__ == "__main__":
    main()