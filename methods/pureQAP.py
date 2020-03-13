import numpy as np
import gc

from problems.placement import PlacementQAP
from methods.exterior_penalty import ExteriorPenaltyMethod
from ports.classical_simanneal import ClassicalNeal

class PureQAP:
    def __init__(self, F, D):
        gc.collect()
        self.F = F
        self.D = D

        self.timing = 0

    def get_timing(self):
        return self.timing

    def run(self):
        print("start running pure QAP.")
        size = self.F.shape[0]
        problem = PlacementQAP(
            size,
            size,
            self.F,
            self.D,
            weight0=50000,
            alpha0=1.1,
            const_weight_inc=True,
            initial_weight_estimate=True
        )
        
        solver = ClassicalNeal()
        method = ExteriorPenaltyMethod(
            problem,
            solver,
            1000
        )

        solution = PlacementQAP.solution_matrix(
            (method.run())[0],
            size,
            size
        )
        self.timing = method.get_timing()
        
        return solution