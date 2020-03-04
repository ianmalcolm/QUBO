import numpy as np
from problems.placement import PlacementQAP
from ports.classical_simanneal import ClassicalNeal
from ports.da.da_solver import DASolver
import qaplib.readqaplib as qaplib
from methods.exterior_penalty import ExteriorPenaltyMethod

import os

def get_canonical_weights():
    ret = {}
    with open('canonical_weights.txt','r') as f:
        for line in f.read().splitlines():
            info = line.split()
            ret[info[0]] = float(info[1])
    return ret

canonical_weights = get_canonical_weights()
for filename in os.listdir('qaplib'):
    with open("canonical_weights.txt", 'a+') as f:
        if filename.endswith('.dat') and filename not in canonical_weights:
            print(filename)
            F, D = qaplib.readqaplib(os.path.join('qaplib',filename))
            size = F.shape[0]
            solver = DASolver()
            weight0 = size * size * 25600 / (30*30)
            problem = PlacementQAP(
                size,
                size,
                F,
                D,
                weight0=weight0,
                alpha0=2,
                const_weight_inc=True
            )
            method = ExteriorPenaltyMethod(problem,solver,1000)
            method.run()
            
            result_string = filename + " " + str(np.max(problem.ms)) + " " + str(method.get_timing()) + '\n'
            f.write(result_string)