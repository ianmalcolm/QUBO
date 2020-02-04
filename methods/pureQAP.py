import numpy as np

from problems.placement import PlacementQAP
from methods.exterior_penalty import ExteriorPenaltyMethod
from ports.classical_simanneal import ClassicalNeal

class PureQAP:
    def __init__(self, F, D):
        self.F = F.copy()
        self.D = D.copy()
    
    def run(self):
        size = self.F.shape[0]
        problem = PlacementQAP(
            size,
            size,
            self.F,
            self.D,
            weight0=50,
            alpha0=2
        )
        
        solver = ClassicalNeal()
        method = ExteriorPenaltyMethod(
            problem,
            solver,
            100
        )

        solution = PlacementQAP.solution_matrix(
            (method.run())[0],
            size,
            size
        )
        return solution