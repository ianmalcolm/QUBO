import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
import json

from problems.placement import PlacementQAP
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
ORDER_DIRNAME = 'orders'
CONFIG_DIRNAME = 'configs'

group_num_cols = 2
group_num_rows = 4
group_num_locs = group_num_cols * group_num_rows

NUM_ITERATIONS = 10
F = None
D = None
TAKE = ['order_3600_300_b.txt']
# perm_file = 'perm270'
def main():
    for filename in os.listdir(ORDER_DIRNAME):
        if filename in TAKE:
            order_character = filename.split('_')
            num_items = num_locs = int(order_character[1])
            num_skus = int(order_character[2])
            
            config_filename = os.path.join(CONFIG_DIRNAME, 'config'+str(num_items)+'_'+str(num_skus)+'.json')
            with open(config_filename, 'r') as f:
                warehouse_config = json.load(f)
            
            for i in range(3):
                result_dict_list = run(filename, warehouse_config)
                result_file = os.path.join(RESULT_FOLDER, filename+'_competitors.csv')
                if os.path.exists(result_file):
                    df = pd.read_csv(result_file,index_col=0)
                else:
                    df = pd.DataFrame()
                
                new_result = postprocess(result_dict_list)
                df = df.append(new_result, ignore_index=True)
                df.to_csv(result_file)
def make_matrix(perm):
    n = len(perm)
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][perm[i]] = 1
    return matrix

def run(order_filename,config):
    NUM_SKUS = int(config['NUM_SKUS'])
    WAREHOUSE_NUM_COLS = int(config['WAREHOUSE_NUM_COLS'])
    WAREHOUSE_NUM_ROWS = int(config['WAREHOUSE_NUM_ROWS'])
    NUM_LOCS = int(config['NUM_LOCS'])
    NUM_GROUPS = int(config['NUM_GROUPS'])
    DIST_VERTICAL = int(config['DIST_VERTICAL'])
    DIST_HORIZONTAL = int(config['DIST_HORIZONTAL'])
    GROUP_NUM_ROWS = int(config['GROUP_NUM_ROWS'])
    GROUP_NUM_COLS = int(config['GROUP_NUM_COLS'])
    EXHAUST_AGGREGATES = config['EXHAUST_AGGREGATES']
    RANDOM_BUNCHING = config['RANDOM_BUNCHING']
    RANDOM_GROUPING = config['RANDOM_GROUPING']

    order_path = os.path.join(ORDER_DIRNAME,order_filename)
    order_parser = OrderParser(order_path, NUM_SKUS, threshold=0)
    global F
    global D
    
    if F is None:
        F = order_parser.gen_F()
    qty = order_parser.summary()

    if D is None:
        D_gen = DistanceGenerator(
            WAREHOUSE_NUM_ROWS,
            WAREHOUSE_NUM_COLS, 
            DIST_VERTICAL, 
            DIST_HORIZONTAL,
            GROUP_NUM_ROWS,
            GROUP_NUM_COLS
            )
        D = D_gen.gen_S_shape()

    order_parser = OrderParser(order_path, NUM_SKUS, threshold=0)
    order_set = order_parser.gen_raw_orders()

    print(qty)
    evaluator = RouteEvaluator(
        qty,
        order_set,
        WAREHOUSE_NUM_COLS,
        WAREHOUSE_NUM_ROWS,
        DIST_VERTICAL,
        3,
        NUM_LOCS,
        NUM_LOCS
    )
    result_dict = {}

    random = RandomMethod(NUM_LOCS, NUM_LOCS)
    sol_random = random.run()
    res_random = evaluator.run(sol_random)
    result_dict['random'] = res_random

    # permutation = []
    # with open(perm_file, 'r') as f:
    #     permutation_string = f.read()
    #     permutation = np.fromstring(permutation_string[1:-1], sep=' ', dtype=np.int32)
    # res_direct = evaluator.run(make_matrix(permutation))

    # abc = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D_euclidean), 8)
    # sol_abc = abc.run()
    # res_abc = evaluator.run(sol_abc)
    # str_abc = "result of abc is " + str(res_abc) + '\n'
    # print(str_abc)
    # result_dict['abc'] = res_abc

    # coi = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D), NUM_LOCS)
    # sol_coi = coi.run()
    # res_coi = evaluator.run(sol_coi)
    # str_coi = "result of coi is " + str(res_coi) + '\n'
    # print(str_coi)
    # result_dict['coi'] = res_coi

    ifhoos = IFHOOS(F,D, beta=0.6)
    sol_ifhoos = ifhoos.run()
    if not all(PlacementQAP.check_mtx(sol_ifhoos)):
        raise ValueError("Unfeasible solution from ifhoos")
    res_ifhoos = evaluator.run(sol_ifhoos)
    str_ifhoos = "result of ifhoos is " + str(res_ifhoos) + '\n'
    print(str_ifhoos)
    result_dict['ifhoos'] = res_ifhoos


    # result_dict['directqap'] = res_direct
    return [result_dict]

def postprocess(result_dict):
    return pd.DataFrame(result_dict)

if __name__ == "__main__":
    main()
