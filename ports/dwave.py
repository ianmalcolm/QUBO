from .solver import Solver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

import dimod
import utils.index as idx
import utils.mtx as mt
import numpy as np

class Dwave(Solver):
    def __init__(self):
        print("Dwave solver created...")
        pass

    def solve(self, matrix, initial={}):
        '''
            returns: a solution
                    solution is a tuple (dict, float) representing sample and energy.
        '''
        if bool(initial):
            initial_sample = dimod.as_samples(initial)
        print("solver starts the process...")
        mtx = matrix.copy()
        print("matrix has %d zeros out of %d" % (np.count_nonzero(mtx==0), mtx.shape[0]*mtx.shape[1]))
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(mtx)

        print("Solver engages Dwave quantum hardware!")
        sampler = dimod.ScaleComposite(EmbeddingComposite(DWaveSampler(num_qubits__gt=2000)))
        
        if bool(initial):
            response = sampler.sample(bqm, initial_states=dimod.SampleSet.from_samples(initial_sample))
        else:
            response = sampler.sample(bqm)
            
        for datum in response.data(fields=['sample','energy','num_occurrences']):
            print(datum)

        return (response.first.sample, response.first.energy)