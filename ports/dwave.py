from .solver import Solver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

import dimod
import utils.index as idx
import utils.mtx as mt
import numpy as np
import itertools

class Dwave(Solver):
    def __init__(self):
        print("Dwave solver created...")
        self.timing = 0

    def get_timing(self):
        return self.timing

    def solve(self, matrix, initial=()):
        '''
            returns: a solution
                    solution is a tuple (dict, float) representing sample and energy.
        '''
        if bool(initial):
            initial_sample = list(initial[0].values())
            print("initial sample: ",initial_sample)
        print("solver starts the process...")
        mtx = matrix.copy()
        print("converting to upper triangular...")
        mtx = mt.to_upper_triangular(mtx)
        print("matrix has %d zeros out of %d" % (np.count_nonzero(mtx==0), mtx.shape[0]*mtx.shape[1]))

        np.savetxt("mtx.txt", mtx, fmt='%d')

        Q = {}
        size = mtx.shape[0]
        for i,j in itertools.product(range(size),range(size)):
            if i==j:
                if mtx[i][i]:
                    Q[(i,i)] = mtx[i][i]
            elif i<j:
                if mtx[i][j]:
                    Q[(i,j)] = mtx[i][j]
            else:
                pass
        
        print("Solver engages Dwave quantum hardware!")
        sampler = dimod.ScaleComposite(EmbeddingComposite(DWaveSampler()))

        anneal_schedule = [(0,1),(4,0.5),(14,0.5),(20,1)]
        if bool(initial):
            print(initial)
            initial_state_dict = initial[0]
            response = sampler.sample_qubo(Q,num_reads=50, anneal_schedule=anneal_schedule, initial_state=initial_state_dict)
        else:
            response = sampler.sample_qubo(Q,num_reads=50, annealing_time=20)

        timing_iter = (response.info['timing']['qpu_sampling_time'] / 1000000)
        self.timing += timing_iter
        print(timing_iter)
        print(response.info)
        for datum in response.data(fields=['sample','energy','num_occurrences']):
            print(datum)
            break

        return (response.first.sample, response.first.energy)