import random
import numpy as np
from simanneal.simanneal import Annealer
import time

class QMKP(Annealer):
    def __init__(self, F, k):
        self.F = F
        self.k = k
        self.n = F.shape[0]
        self.s = int(self.n / self.k)
        self.timing = 0
        state = QMKP.initialize_state(self.n, self.k)
        super(QMKP, self).__init__(state)

    @staticmethod
    def initialize_state(n, k):
        '''Generate a random grouping'''
        state = np.zeros(shape=n, dtype=np.int8)
        s = int(n/k)        
        a=0
        for i in range(k):
            for j in range(s):
                state[a]=i
                a+=1
        random.shuffle(state)
        print("after shuffle",state)
        return state

    def move(self):
        ''' swaps i,j of group a,b to group b,a'''

        a = random.randint(0, self.state.shape[0]-1)
        b = random.randint(0, self.state.shape[0]-1)

        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        '''calculates interaction frequency within subsets'''
        # scans state to record subset members
        members = {}
        for i in range(self.k):
            members[i] = []
        for i in range(self.n):
            members[self.state[i]].append(i)
        
        # computes the sum of all internal interaction freq
        energy = 0
        for i in range(self.k):
            subset = members[i]
            for p in range(self.s):
                for q in range(p, self.s):
                    energy += self.F[subset[p]][subset[q]]
        return energy
    
    @staticmethod
    def solution_matrix(state, n,k):
        matrix = np.zeros(shape=(n,k), dtype=np.int8)
        for i in range(n):
            matrix[i][state[i]] = 1
        return matrix