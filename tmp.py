import qaplib.readqaplib as qaplib
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

F,D = qaplib.readqaplib('order_180_30_a.txt.dat')

# with open('tai35b.dat', 'w') as f:
#     f.write(format(F,D))

df = pd.read_csv("simdata_heu_da/order_180_30_a.txt.csv", index_col=0)
permutation_string = df['perm'][0]
print(permutation_string)
print(type(permutation_string))
permutation = np.fromstring(permutation_string[1:-1], sep=' ')
print(permutation)

ans = np.random.permutation(180)

evaluator = QAPEvaluator(
    180,180,F,D
)
# evaluator1 = QAPEvaluator(
#     144,144,F1,D1
# )
print(std.obj(F,D,ans))
# print(std.obj(F1,D1,ans))
print(evaluator.run(make_matrix(permutation.astype(np.int32))))
# print(evaluator1.run(make_matrix(ans)))

