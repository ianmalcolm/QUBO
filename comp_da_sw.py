from ports.da.da_solver import DASolver
from ports.classical_simanneal import ClassicalNeal
from problems.placement import PlacementQAP
import qaplib.readqaplib as data
from sim.test_qap import QAPEvaluator
from methods.exterior_penalty import ExteriorPenaltyMethod

import numpy as np
import os

def main():
    da_solver = DASolver()
    simanneal_solver = ClassicalNeal()
    F, D = data.readqaplib(os.path.join("qaplib","sko42.dat"))
    size = F.shape[0]

    problem = PlacementQAP(
        size,
        size,
        F,
        D,
        weight0=5,
        alpha0=5,
        const_weight_inc=False
    )
    method = ExteriorPenaltyMethod(problem, da_solver, 10000)
    method.run()
    test_mtx = problem.flow + problem.cts[2]
    da_ans=da_solver.solve(test_mtx,test_mode=True)
    for answer in da_ans:
        solution_dict = answer[0]
        sample_solution_mtx = PlacementQAP.solution_matrix(da_ans[0][0], size,size)
        if all(problem.check_mtx(sample_solution_mtx)):
            print(solution_dict, answer[1])

    input()
    sw_ans=simanneal_solver.solve(test_mtx,test_mode=True)
if __name__ == "__main__":
    main()