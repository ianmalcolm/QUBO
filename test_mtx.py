import numpy as np
from ports.classical_simanneal import ClassicalNeal
from ports.dwave import Dwave
from ports.da.da_solver import DASolver
from problems.placement import PlacementQAP

mtx = np.loadtxt("mtx.txt")

port = DASolver()
solution = port.solve(mtx)
input()

port = ClassicalNeal()
solution = port.solve(mtx)
solution_mtx = PlacementQAP.solution_matrix(solution[0], 8,8)
np.set_printoptions(threshold=np.inf)
print(solution_mtx)
input()
port = Dwave()
solution = port.solve(mtx)
solution_mtx = PlacementQAP.solution_matrix(solution[0], 8,8)
np.set_printoptions(threshold=np.inf)
print(solution_mtx)