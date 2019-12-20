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

        print(mtx.shape)
        print(np.count_nonzero(mtx))

        linear = {}
        quadratic = {}

        print("solver is preparing coefficient dict.")
        for i in range(size):
            for j in range(size):
                #lower triangular part is thrown away
                if i == j:
                    linear[i] = mtx[i][i]
                elif i<j and (not mtx[i][j]==0):
                    quadratic[(i,j)] = mtx[i][j]
        print("solver is constructing bqm.")
        with open("quadratic.dat", 'w') as f: 
            for k,v in quadratic.items():
                f.write(str(k) + ',' + str(v) +'\n')
        bqm = dimod.BQM(
            linear = linear,
            quadratic = quadratic,
            offset = 0.0,
            vartype = dimod.BINARY
        )
        input()

        print("Solver engages Dwave quantum hardware!")
        sampler = dimod.ScaleComposite(EmbeddingComposite(DWaveSampler(num_qubits__gt=2000)))
        response = sampler.sample(bqm)
        for sample, energy, num_occurrences in response.data():
            print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)

        return (response.first.sample, response.first.energy)