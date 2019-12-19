from .solver import Solver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, ScaleComposite

import dimod
import utils.index as idx
import numpy as np

class Dwave(Solver):
    def __init__(self):
        print("Dwave solver created...")
        pass

    def solve(self, matrix):
        '''
            returns: a solution
                    solution is a tuple (dict, float) representing sample and energy.
        '''
        print("solver starts the process...")
        mtx = matrix.copy()
        var_id = 'x'
        size = mtx.shape[0]
        #process matrix to be only upper-triangular significant
        mtx = mtx + np.transpose(mtx)
        for i in range(size):
            mtx[i][i] = mtx[i][i] / 2

        linear = {}
        quadratic = {}

        print("solver is preparing coefficient dict.")
        for i in range(size):
            for j in range(size):
                #lower triangular part is anulled
                if i == j:
                    linear[i] = mtx[i][i]
                elif i<j:
                    quadratic[(i,j)] = mtx[i][j]
        print("solver is constructing bqm.")
        for k,v in linear.items():
            print((k,v))
        with open("quadratic.dat", 'w') as f: 
            for k,v in quadratic.items():
                f.write(str(k) + ',' + str(v) +'\n')
        bqm = dimod.BQM(
            linear = linear,
            quadratic = quadratic,
            offset = 0,
            vartype = dimod.BINARY
        )
        print("Solver engages Dwave quantum hardware!")
        sampler = ScaleComposite(EmbeddingComposite(DWaveSampler()))
        response = sampler.sample(bqm)
        for sample, energy, num_occurrences in response.data():
            print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)

        return (response.first.sample, response.first.energy)