import numpy as np
from problems.placement import PlacementQAP
from ports.classical_simanneal import ClassicalNeal
from ports.da.da_solver import DASolver
import qaplib.readqaplib as qaplib
from methods.exterior_penalty import ExteriorPenaltyMethod

import os

f = open("canonical_weights.txt", 'w')
for filename in os.listdir('qaplib'):
    if filename.endswith('.dat'):
        F, D = qaplib.readqaplib(os.path.join('qaplib',filename))
        size = F.shape[0]
        solver = DASolver()
        problem = PlacementQAP(
            size,
            size,
            F,
            D,
            weight0=200,
            alpha0=2,
            const_weight_inc=True
        )
        method = ExteriorPenaltyMethod(problem,solver,1000)
        method.run()
        
        result_string = filename + " " + str(np.average(problem.ms)) + " " + str(method.get_timing()) + '\n'
        f.write(result_string)