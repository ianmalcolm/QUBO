import os
import math
import json
import numpy as np
import pandas as pd
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
from sim.test_qap import QAPEvaluator

import utils.mtx as mtx

RESULT_FOLDER = "simdata_heu_da"
ORDER_DIRNAME = 'orders_heu_da'
CONFIG_DIRNAME = 'configs'

TAKE = ['order_270_30_a.txt']

def main():
    for filename in os.listdir(ORDER_DIRNAME):
        if filename in TAKE:
            print(filename)
            order_character = filename.split('_')
            num_items = num_locs = int(order_character[1])
            num_skus = int(order_character[2])
            
            config_filename = os.path.join(CONFIG_DIRNAME, 'config'+str(num_items)+'_'+str(num_skus)+'.json')
            with open(config_filename, 'r') as f:
                warehouse_config = json.load(f)

            for i in range(1):
                result_dict_list = run(filename, warehouse_config)
            
                result_filename = filename + ".csv"
                if os.path.exists(os.path.join(RESULT_FOLDER, result_filename)):
                    df = pd.read_csv(os.path.join(RESULT_FOLDER, result_filename),index_col=0)
                else:
                    df = pd.DataFrame()
                new_results=postprocess(result_dict_list)
                df = df.append(new_results, ignore_index=True)
                df.to_csv(os.path.join(RESULT_FOLDER,result_filename))

def run(order_filename, config):
    order_path = os.path.join(ORDER_DIRNAME, order_filename)
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

    order_parser = OrderParser(order_path, NUM_SKUS, threshold=0)
    order_set = order_parser.gen_raw_orders()
    # F: (n by n) symmetric interaction frequency matrix
    F = order_parser.gen_F()
    print("F has shape: ", F.shape)
    
    # D: (m by m) symmetric distance matrix
    D_gen = DistanceGenerator(
        WAREHOUSE_NUM_ROWS,
        WAREHOUSE_NUM_COLS,
        DIST_VERTICAL,
        DIST_HORIZONTAL,
        GROUP_NUM_ROWS,
        GROUP_NUM_COLS
        )
    D = D_gen.gen_S_shape()

    qty = order_parser.summary()
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
    evaluator_qap = QAPEvaluator(
        NUM_LOCS,
        NUM_LOCS,
        F,
        D
    )
    res_dict = {}
    
    exhaust_permutation=False
    if EXHAUST_AGGREGATES=='y':
        exhaust_permutation=True
    random_bunching=False
    if RANDOM_BUNCHING=='y':
        random_bunching=True
    random_grouping=False
    if RANDOM_GROUPING=='y':
        random_grouping=True

    heuristic_da = OurHeuristic(
        NUM_LOCS,
        NUM_LOCS,
        NUM_GROUPS,
        F,
        D,
        fine_weight0=40000,
        fine_alpha0=0,
        num_rows=WAREHOUSE_NUM_ROWS,
        num_cols=WAREHOUSE_NUM_COLS,
        const_weight_inc=True,
        use_dwave_da_sw='da',
        random_bunching=random_bunching,
        random_grouping=random_grouping,
        exhaust_permutation=exhaust_permutation
    )
    sols_heuristic_da = heuristic_da.run()
    res_list = []
    loop_index = 0
    for sol_heuristic_da in sols_heuristic_da:
        res_dict = {}
        is_canonical = heuristic_da.canonical_record[loop_index]
        t_heuristic_da = heuristic_da.get_timing()
        res_heuristic_da = evaluator.run(sol_heuristic_da)
        qapres_heuristic_da = evaluator_qap.run(sol_heuristic_da)
        res_dict['is_canonical']= is_canonical
        res_dict['random_bunching']=heuristic_da.random_bunching
        res_dict['random_grouping']=heuristic_da.random_grouping
        res_dict['qapres_heu_da']= qapres_heuristic_da
        res_dict['res_heu_da']= res_heuristic_da
        timing_list = t_heuristic_da['overall'] + t_heuristic_da['partition'][loop_index]
        res_dict['time_heu_da']= timing_list
        res_dict['perm']=mtx.make_perm(sol_heuristic_da)
        res_list.append(res_dict)
        loop_index += 1
    return res_list

def postprocess(result_dict):
    return pd.DataFrame(result_dict)

def random_filename():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "res_"+current_time+".txt"

if __name__ == "__main__":
    main()
