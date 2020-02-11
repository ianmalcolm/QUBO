import numpy as np

from problems.placement import PlacementQAP
from methods.exterior_penalty import ExteriorPenaltyMethod
from ports.classical_simanneal import ClassicalNeal

class PureQAP:
    def __init__(self, F, D):
        self.F = F.copy()
        self.D = D.copy()

        self.timing = 0

    def get_timing(self):
        return self.timing

    def run(self):
        size = self.F.shape[0]
        problem = PlacementQAP(
            size,
            size,
            self.F,
            self.D,
            weight0=80,
            alpha0=10
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