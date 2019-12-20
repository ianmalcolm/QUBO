from .solver import Solver
import neal

import dimod
import utils.index as idx
import utils.mtx as mt
import numpy as np

class ClassicalNeal(Solver):
    def __init__(self):
        print("Classical simanneal solver created...")
        pass

    def solve(self, matrix):
        '''
            returns: a solution
                    solution is a tuple (dict, float) representing sample and energy.
        '''
        print("solver starts the process...")
        mtx = matrix.copy()
        print("matrix has %d zeros out of %d" % (np.count_nonzero(mtx==0), mtx.shape[0]*mtx.shape[1]))
        print("Constructing bqm out of matrix...")
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(mtx)
        input("bqm constructed. press any key to continue.")
        print("Classical simanneal starts now!")
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample(bqm)
        for datum in response.data(fields=['sample','energy','num_occurrences']):
            print(datum)

        return (response.first.sample, response.first.energy)