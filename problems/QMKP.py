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
        item_state = np.zeros(shape=n, dtype=np.int8)
        s = int(n/k)        
        a=0
        for i in range(k):
            for j in range(s):
                item_state[a]=i
                a+=1
        random.shuffle(item_state)
        print("after shuffle",item_state)
        
        # prepare a rec of where has what
        subset_state = []
        for i in range(k):
            subset_state.append([])
        for i in range(n):
            subset_state[item_state[i]].append(i)
        return (item_state, subset_state)

    def move(self):
        ''' swaps i,j of group a,b to group b,a'''
        # e0 = self.energy()
        item_state, subset_state = self.state

        a = random.randint(0, self.n-1)
        b = random.randint(0, self.n-1)
        
        prev_interactions = 0
        subset_a = subset_state[item_state[a]]
        subset_b = subset_state[item_state[b]]
        if subset_a == subset_b:
            return 0
        # print("a: ", a)
        # print("b: ", b)
        # print("subset a: ",subset_a)
        # print("subset b: ",subset_b)
        for i in range(self.s):
            if subset_a[i] != a:
                prev_interactions += self.F[subset_a[i]][a]
            if subset_b[i] != b:
                prev_interactions += self.F[subset_b[i]][b]

        new_interactions = 0
        for i in range(self.s):
            if subset_a[i] != a:
                new_interactions += self.F[subset_a[i]][b]
            if subset_b[i] != b:
                new_interactions += self.F[subset_b[i]][a]

        dE = new_interactions - prev_interactions

        # record update
        subset_state[item_state[a]].remove(a)
        subset_state[item_state[a]].append(b)
        subset_state[item_state[b]].remove(b)
        subset_state[item_state[b]].append(a)
        item_state[a], item_state[b] = item_state[b], item_state[a]

        # e1 = self.energy()
        # de1 = e1-e0
        # print("de1 from inside: ", de1)
        # print("de from inside: ", dE)
        # print('e1 from inside: ', e1)
        # print('e0 from inside: ', e0)
        # return difference in energy
        return dE

    def energy(self):
        '''calculates interaction frequency within subsets'''
        # read subset members
        members = self.state[1]
        
        # computes the sum of all internal interaction freq
        energy = 0
        for i in range(self.k):
            subset = members[i]
            for p in range(self.s):
                for q in range(p+1, self.s):
                    energy += self.F[subset[p]][subset[q]]
        return energy
    
    @staticmethod
    def solution_matrix(state, n,k):
        item_state = state[0]
        matrix = np.zeros(shape=(n,k), dtype=np.int8)
        for i in range(n):
            matrix[i][item_state[i]] = 1
        return matrix