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

RESULT_FOLDER = "simdata_pure"
ORDER_DIRNAME = 'orders_pure'
CONFIG_DIRNAME = 'configs'

def main():
    for filename in os.listdir(ORDER_DIRNAME):
        if filename.endswith('.txt'):
            print(filename)
            order_character = filename.split('_')
            num_items = num_locs = int(order_character[1])
            num_skus = int(order_character[2])
            
            config_filename = os.path.join(CONFIG_DIRNAME, 'config'+str(num_items)+'_'+str(num_skus)+'.json')
            with open(config_filename, 'r') as f:
                warehouse_config = json.load(f)

            result_dict_list = []
            for i in range(5):
                result_dict = run(filename, warehouse_config)
                result_dict_list.append(result_dict)
            
            result_filename = filename + ".csv"
            if os.path.exists(result_filename):
                df = pd.read_csv(result_filename)
            else:
                df = pd.DataFrame()
            new_results=postprocess(result_dict_list)
            df.append(new_results, ignore_index=True)
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
        DIST_HORIZONTAL,
        NUM_LOCS,
        NUM_LOCS
    )
    evaluator_qap = QAPEvaluator(
        NUM_SKUS,
        NUM_LOCS,
        F,
        D
    )
    res_dict = {}

    pureQAP = PureQAP(
        F,
        D
    )
    sol_pureQAP = pureQAP.run()
    t_pureQAP = pureQAP.get_timing()
    res_pure = evaluator.run(sol_pureQAP)
    qapres_pure = evaluator_qap.run(sol_pureQAP)
    res_dict['qapres_pure'] = qapres_pure
    res_dict['res_pure']= res_pure
    res_dict['time_pure'] = t_pureQAP

    return res_dict

def postprocess(result_dict):
    return pd.DataFrame(result_dict)

def random_filename():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "res_"+current_time+".txt"

if __name__ == "__main__":
    main()
