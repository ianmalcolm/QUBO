from .solver import Solver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import utils.index as idx
import numpy as np

class Dwave(Solver):
    def __init__(self, mtx, num_ancillaries=0):
        self.mtx = mtx.copy()
        self.num_ancillaries = num_ancillaries
    
    def solve(self):
        var_id = 'x'
        ancillary_id = 'a'
        size = self.mtx.shape[0]
        #process matrix to be only upper-triangular significant
        self.mtx = self.mtx + np.transpose(self.mtx)
        for i in range(size):
            self.mtx[i][i] = self.mtx[i][i] / 2

        linear = {}
        quadratic = {}

        for i in range(size):
            for j in range(size):
                if i >= size-self.num_ancillaries:
                    id_i = idx.var_str(ancillary_id,i)
                else:
                    id_i = idx.var_str(var_id,i)
                if j >= size-self.num_ancillaries:
                    id_j = idx.var_str(ancillary_id,j)
                else:
                    id_j = idx.var_str(var_id,j)
                
                #only process upper triangular part
                if i == j:
                    linear[id_i] = self.mtx[i][i]
                elif i<j:
                    quadratic[(id_i, id_j)] = self.mtx[i][j]
            
        Q = dict(linear)
        Q.update(quadratic)
        response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=1000)
        for sample, energy, num_occurrences in response.data():
            print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)