import numpy as np
from ports.classical_simanneal import ClassicalNeal
from problems.placement import PlacementQAP

mtx = np.loadtxt("mtx.txt")

port = ClassicalNeal()
solution = port.solve(mtx)
solution_mtx = PlacementQAP.solution_matrix(solution[0], 8,8)