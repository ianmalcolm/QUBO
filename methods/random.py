import numpy as np

class RandomMethod:
    def __init__(self, n, m):
        self.n = n
        self.m = m
    def run(self):
        ret = np.zeros((self.n, self.m))
        perm = np.random.permutation(self.n)
        for i in range(self.n):
            ret[i][perm[i]] = 1
        return ret