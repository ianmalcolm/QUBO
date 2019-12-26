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

    def solve(self, matrix, initial=()):
        '''
            returns: a solution
                    solution is a tuple (dict, float) representing sample and energy.
        '''
        print("solver starts the process...")
        if bool(initial):
            initial_sample = dimod.as_samples(initial[0])
        mtx = matrix.copy()
        print("matrix has %d zeros out of %d" % (np.count_nonzero(mtx==0), mtx.shape[0]*mtx.shape[1]))
        print("Constructing bqm out of matrix...")
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(mtx)
        print("Classical simanneal starts now!")
        sampler = neal.SimulatedAnnealingSampler()
        
        if bool(initial):
            response = sampler.sample(bqm, initial_states=dimod.SampleSet.from_samples(initial_sample, vartype='BINARY', energy=[initial[1]]),num_reads=100,initial_states_generator='tile')
        else:
            response = sampler.sample(bqm, num_reads=100)

        for datum in response.data(fields=['energy','num_occurrences']):
            print(datum)
        
        return (response.first.sample, response.first.energy)