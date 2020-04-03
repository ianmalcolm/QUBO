import qaplib.readqaplib as qaplib
from gen_qap_dat import format
import numpy as np
import sim.std_qap as std
from sim.test_qap import QAPEvaluator
from DistanceGenerator import DistanceGenerator
from problems.placement import PlacementQAP

import pandas as pd
from orders.order_parser import OrderParser

def make_matrix(perm):
    n = len(perm)
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][perm[i]] = 1
    return matrix

# order_parser = OrderParser(
#     'orders/order_144_30_a.txt',
#     30,
#     0
# )
# F1 = order_parser.gen_F()
# D_gen = DistanceGenerator(
#     6,
#     24,
#     1,
#     3,
#     None,
#     None
# )
# D1 = D_gen.gen_S_shape()

F,D = qaplib.readqaplib('order_8100_500_a.txt.dat')

# with open('tai35b.dat', 'w') as f:
#     f.write(format(F,D))

df = pd.read.csv("simdata_heu_da/order_8100_500_a.txt.csv.back", index_col=0)
permutation_string = df['perm']
print(permutation_string)
input()
    


ans = np.random.permutation(90)

evaluator = QAPEvaluator(
    90,90,F,D
)
# evaluator1 = QAPEvaluator(
#     144,144,F1,D1
# )
print(std.obj(F,D,ans))
# print(std.obj(F1,D1,ans))
print(evaluator.run(make_matrix(ans)))
# print(evaluator1.run(make_matrix(ans)))
