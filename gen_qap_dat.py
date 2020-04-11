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

ORDER_DIRNAME = 'orders'
CONFIG_DIRNAME = 'configs'
TAKE = ['order_270_30_b.txt']

# prepares order data to file
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
            
            run(filename, warehouse_config)

def run(order_filename, config):
    NUM_SKUS = int(config['NUM_SKUS'])
    WAREHOUSE_NUM_COLS = int(config['WAREHOUSE_NUM_COLS'])
    WAREHOUSE_NUM_ROWS = int(config['WAREHOUSE_NUM_ROWS'])
    DIST_VERTICAL = int(config['DIST_VERTICAL'])
    DIST_HORIZONTAL = int(config['DIST_HORIZONTAL'])
    GROUP_NUM_ROWS = int(config['GROUP_NUM_ROWS'])
    GROUP_NUM_COLS = int(config['GROUP_NUM_COLS'])

    order_path = os.path.join(ORDER_DIRNAME, order_filename)
    order_parser = OrderParser(order_path, NUM_SKUS, threshold=0)
    # F: (n by n) symmetric interaction frequency matrix
    F = order_parser.gen_F()
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
    with open(order_filename+'.dat', 'w') as f:
        f.write(format(F,D))


def format(F,D):
    size = F.shape[0]
    ret = str(size) + '\n\n'
    for i in range(size):
        row = F[i,:]
        row_string = " ".join(map(str,row)) + '\n'
        ret += row_string

    ret += '\n'
    for i in range(size):
        row = D[i,:]
        row_string = " ".join(map(str,row)) + '\n'
        ret += row_string
    
    return ret

if __name__ == "__main__":
    main()
