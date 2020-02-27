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
    D_euclidean = D_gen.gen_Euclidean()
    np.set_printoptions(threshold=np.inf)
    print("D: ", D)
    np.set_printoptions(threshold=6)

    qty = order_parser.summary()
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
    res_list = []
    time_list = []

    USE_DWAVE = input("use heuristic w dwave?")
    USE_PURE = input("use pure QAP?")
    USE_DA = input("use da?")

######################################################################
    ifhoos = IFHOOS(F,D)
    sol_ifhoos = ifhoos.run()
    res_ifhoos = evaluator.run(sol_ifhoos)
    res_list.append(str(res_ifhoos))
    
    if USE_DA=="y":
        heuristic_da = OurHeuristic(
            NUM_LOCS,
            NUM_LOCS,
            NUM_GROUPS,
            F,
            D,
            DIST_HORIZONTAL,
            WAREHOUSE_NUM_ROWS,
            WAREHOUSE_NUM_COLS,
            fine_weight0=30000,
            fine_alpha0=1000,
            const_weight_inc=False,
            use_dwave_da_sw='da'
        )
        sol_heuristic_da = heuristic_da.run()
        t_heuristic_da = heuristic_da.get_timing()
        res_heuristic_da = evaluator.run(sol_heuristic_da)
        res_list.append(str(res_heuristic_da))
        time_list.append(str(t_heuristic_da))

    if USE_PURE=="y":
        heuristic_qpu = OurHeuristic(
            NUM_LOCS,
            NUM_LOCS,
            NUM_GROUPS,
            F,
            D,
            DIST_HORIZONTAL,
            WAREHOUSE_NUM_ROWS,
            WAREHOUSE_NUM_COLS,
            fine_weight0=40000,
            fine_alpha0=0,
            const_weight_inc=False,
            use_dwave_da_sw='dwave'
        )
        sol_heuristic_qpu = heuristic_qpu.run()
        t_heuristic_qpu = heuristic_qpu.get_timing() 
        res_heuristic_qpu = evaluator.run(sol_heuristic_qpu)
        res_list.append(str(res_heuristic_qpu))
        time_list.append(str(t_heuristic_qpu))

    heuristic_sw = OurHeuristic(
        NUM_LOCS,
        NUM_LOCS,
        NUM_GROUPS,
        F,
        D,
        DIST_HORIZONTAL,
        WAREHOUSE_NUM_ROWS,
        WAREHOUSE_NUM_COLS,
        fine_weight0=40000,
        fine_alpha0=0,
        const_weight_inc=False,
        use_dwave_da_sw='sw'
    )
    sol_heuristic_sw = heuristic_sw.run()
    t_heuristic_sw = heuristic_sw.get_timing()
    res_heuristic_sw = evaluator.run(sol_heuristic_sw)
    res_list.append(str(res_heuristic_sw))
    time_list.append(str(t_heuristic_sw))

    if USE_PURE=='y':
        pureQAP = PureQAP(
            F,
            D
        )
        sol_pureQAP = pureQAP.run()
        t_pureQAP = pureQAP.get_timing()
        res_pure = evaluator.run(sol_pureQAP)
        res_list.append(str(res_pure))
        time_list.append(str(t_pureQAP))

    abc = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D_euclidean), 3)
    sol_abc = abc.run()
    res_abc = evaluator.run(sol_abc)
    res_list.append(str(res_abc))

    coi = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D), NUM_LOCS)
    sol_coi = coi.run()
    res_coi = evaluator.run(sol_coi)
    res_list.append(str(res_coi))

    random = RandomMethod(NUM_LOCS, NUM_LOCS)
    sol_random = random.run()
    res_random = evaluator.run(sol_random)
    res_list.append(str(res_random))


    result_filename = str(NUM_SKUS) + "SKUs_" + str(WAREHOUSE_NUM_ROWS) + "*" + str(WAREHOUSE_NUM_COLS) + ".txt"
    with open(os.path.join(RESULT_FOLDER, result_filename), 'a+') as f:
        res = " ".join(res_list) + '\n'
        f.write(res)

    filename_t = str(NUM_SKUS) + "SKUs_" + str(WAREHOUSE_NUM_ROWS) + "*" + str(WAREHOUSE_NUM_COLS) + "_time.txt"
    with open(os.path.join(RESULT_FOLDER, filename_t), 'a+') as f:
        str_t = " ".join(time_list) + '\n'
        f.write(str_t)
    
def random_filename():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "res_"+current_time+".txt"

if __name__ == "__main__":
    main()