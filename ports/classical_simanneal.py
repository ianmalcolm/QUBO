import time
import itertools
import gc

from .solver import Solver
import neal
import dimod
import utils.index as idx
import utils.mtx as mt
import numpy as np

class ClassicalNeal(Solver):
    def __init__(self):
        print("Classical simanneal solver created...")
        self.timing = 0

    def get_timing(self):
        return self.timing

    def solve(self, matrix, initial=(), test_mode=False):
        '''
            returns: a solution
                    solution is a tuple (dict, float) representing sample and energy.
        '''
        #print("solver starts the process...")
        if bool(initial):
            initial_sample = dimod.as_samples(initial[0])
        mtx = matrix.copy()
        print("Converting matrix to upper triangular...")
        mtx = mt.to_upper_triangular(mtx)
        np.savetxt("mtx.txt", mtx, fmt='%d')
        #np.set_printoptions(threshold=np.inf)
        #print(mtx)
        #np.set_printoptions(threshold=6)

        #print("matrix has %d zeros out of %d" % (np.count_nonzero(mtx==0), mtx.shape[0]*mtx.shape[1]))
        #print("Constructing bqm out of matrix...")
        #bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(mtx)
        #print("Classical simanneal starts now!")
        
        sampler = neal.SimulatedAnnealingSampler()
        
        gc.collect()
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

        params = super().sa_params(mtx)

        start_time = time.time()
        print("ClassicalNeal begins sampling.")
        if bool(initial):
            response = sampler.sample_qubo(
                Q, 
                initial_states=dimod.SampleSet.from_samples(initial_sample, vartype='BINARY', energy=[initial[1]]),
                num_reads=params['number_runs'],
                initial_states_generator='tile',
                num_sweeps=1000            
            )
        else:
            response = sampler.sample_qubo(
                Q,
                num_reads=params['number_runs']            
            )
        end_time = time.time()

        timing_iter = end_time - start_time
        print("one iteration takes ", timing_iter)

        if test_mode:
            self.timing = 0
        self.timing += timing_iter

        for datum in response.data(fields=['energy','num_occurrences']):
            print(datum)
        
        if not test_mode:
            return (response.first.sample, response.first.energy)
        else:
            return self.to_solution_dict(response)
        
    def to_solution_dict(self,response):
        ret = []
        for conf, energy, freq in response.data(fields=['sample','energy','num_occurrences']):
            ret.append((conf, energy, freq))
        return ret