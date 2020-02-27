from ports.da.da_solver import DASolver
import numpy as np

x= np.ones((3,3))

solver = DASolver()
solver.solve(x)