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

import utils.mtx as mtx

RESULT_FOLDER = "simdata"
ORDER_DIRNAME = 'orders'
CONFIG_DIRNAME = 'configs'

def main():
    for filename in os.listdir(ORDER_DIRNAME):
        if filename.endswith('.txt'):
            order_character = filename.split('_')
            num_items = num_locs = int(order_character[1])
            num_skus = int(order_character[2])
            
            config_filename = os.path.join(CONFIG_DIRNAME, 'config'+str(num_items)+'_'+str(num_skus)+'.json')
            with open(config_filename, 'r') as f:
                warehouse_config = json.load(f)

            result_dict_list = []
            for i in range(10):
                result_dict = run(filename, warehouse_config)
                result_dict_list.append(result_dict)
            
            result_filename = filename + ".csv"
            df=postprocess(result_dict_list)
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
    np.set_printoptions(threshold=np.inf)
    print("F: ", F)
    np.set_printoptions(threshold=6)
    
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
    res_dict = {}

    USE_DWAVE = config["USE_DWAVE"]
    USE_PURE = config["USE_PURE"]
    USE_DA = config["USE_DA"]

######################################################################
    ifhoos = IFHOOS(F,D)
    sol_ifhoos = ifhoos.run()
    res_ifhoos = evaluator.run(sol_ifhoos)
    res_dict['res_ifhoos']=res_ifhoos
    
    if USE_DA=="y":
        heuristic_da = OurHeuristic(
            NUM_LOCS,
            NUM_LOCS,
            NUM_GROUPS,
            F,
            D,
            fine_weight0=30000,
            fine_alpha0=1000,
            const_weight_inc=False,
            use_dwave_da_sw='da'
        )
        sol_heuristic_da = heuristic_da.run()
        t_heuristic_da = heuristic_da.get_timing()
        res_heuristic_da = evaluator.run(sol_heuristic_da)
        res_dict['res_heu_da']=res_heuristic_da
        res_dict['t_heu_da']=t_heuristic_da

    if USE_DWAVE=="y":
        heuristic_qpu = OurHeuristic(
            NUM_LOCS,
            NUM_LOCS,
            NUM_GROUPS,
            F,
            D,
            fine_weight0=40000,
            fine_alpha0=0,
            const_weight_inc=False,
            use_dwave_da_sw='dwave'
        )
        sol_heuristic_qpu = heuristic_qpu.run()
        t_heuristic_qpu = heuristic_qpu.get_timing() 
        res_heuristic_qpu = evaluator.run(sol_heuristic_qpu)
        res_dict['res_heu_qpu']= res_heuristic_qpu
        res_dict['t_heu_qpu']= t_heuristic_qpu

    heuristic_sw = OurHeuristic(
        NUM_LOCS,
        NUM_LOCS,
        NUM_GROUPS,
        F,
        D,
        fine_weight0=40000,
        fine_alpha0=0,
        const_weight_inc=False,
        use_dwave_da_sw='sw'
    )
    sol_heuristic_sw = heuristic_sw.run()
    t_heuristic_sw = heuristic_sw.get_timing()
    res_heuristic_sw = evaluator.run(sol_heuristic_sw)
    res_dict['res_heu_sw']= res_heuristic_sw
    res_dict['time_heu_sw']= t_heuristic_sw

    if USE_PURE=='y':
        pureQAP = PureQAP(
            F,
            D
        )
        sol_pureQAP = pureQAP.run()
        t_pureQAP = pureQAP.get_timing()
        res_pure = evaluator.run(sol_pureQAP)
        res_dict['res_pure']= res_pure
        res_dict['time_pure'] = t_pureQAP

    abc = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D_euclidean), 3)
    sol_abc = abc.run()
    res_abc = evaluator.run(sol_abc)
    res_dict['res_abc']= res_abc

    coi = ABCMethod(NUM_LOCS, NUM_LOCS, np.diag(F), np.diag(D), NUM_LOCS)
    sol_coi = coi.run()
    res_coi = evaluator.run(sol_coi)
    res_dict['res_coi']= res_coi

    random = RandomMethod(NUM_LOCS, NUM_LOCS)
    sol_random = random.run()
    res_random = evaluator.run(sol_random)
    res_dict['res_rand']= res_random

    return res_dict

def postprocess(result_dict):
    return pd.DataFrame(result_dict)

def random_filename():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "res_"+current_time+".txt"

if __name__ == "__main__":
    main()