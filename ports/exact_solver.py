from .solver import Solver

import dimod
import utils.index as idx
import utils.mtx as mt
import numpy as np

class ExactSolv(Solver):
    def __init__(self):
        print("Exact solver created...")
        pass

    def solve(self, matrix, initial=()):
        '''
            returns: a solution
                    solution is a tuple (dict, float) representing sample and energy.
        '''
        print("solver starts the process...")
        mtx = matrix.copy()
        print("Converting matrix to upper triangular...")
        mtx = mt.to_upper_triangular(mtx)
        
        #np.set_printoptions(threshold=np.inf)
        #print(mtx)
        np.savetxt("mtx.txt", mtx, fmt='%d')
        np.set_printoptions(threshold=6)

        print("matrix has %d zeros out of %d" % (np.count_nonzero(mtx==0), mtx.shape[0]*mtx.shape[1]))
        print("Constructing bqm out of matrix...")
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(mtx)
        print("Exact solver starts now!")
        sampler = dimod.ExactSolver()
        response = sampler.sample(bqm)

        for datum in response.data(fields=['energy','num_occurrences']):
            print(datum)
        
        return (response.first.sample, response.first.energy)