from .solver import Solver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import utils.index as idx
import numpy as np

class Dwave(Solver):
    def __init__(self):
        pass

    def solve(self, matrix):
        '''
            returns: a solution
                    solution is a tuple (dict, float) representing sample and energy.
        '''
        mtx = matrix.copy()
        var_id = 'x'
        size = mtx.shape[0]
        #process matrix to be only upper-triangular significant
        mtx = mtx + np.transpose(mtx)
        for i in range(size):
            mtx[i][i] = mtx[i][i] / 2

        linear = {}
        quadratic = {}

        for i in range(size):
            for j in range(size):
                id_i = idx.var_str(var_id,i)
                id_j = idx.var_str(var_id,j)
                
                #only process upper triangular part
                if i == j:
                    linear[id_i] = mtx[i][i]
                elif i<j:
                    quadratic[(id_i, id_j)] = mtx[i][j]
            
        Q = dict(linear)
        Q.update(quadratic)
        response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=1000)
        for sample, energy, num_occurrences in response.data():
            print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
        
        return (response.first.sample, response.first.energy)