import numpy as np
import math

class ABCMethod:
    def __init__(self, n,m, popularity, distance, num_grps):
        '''
            parameters:
            popularity      popularity array, indexed by item index
            distance        distance array, indexed by location index
            num_grps        number of popularity classes for items
                NOTE: when num_grps = n, this method becomes COI
        '''
        self.n = n
        self.m = m

        indices_item = np.arange(0,n)
        dtype_pop = [('idx',int),('pop',int)]
        self.popularity = np.sort(np.array(list(zip(indices_item,popularity.copy())),dtype=dtype_pop), order='pop')[::-1]

        indices_location = np.arange(0,m)
        dtype_dist = [('idx',int),('dist',int)]
        self.distance = np.sort(np.array(list(zip(indices_location,distance.copy())),dtype=dtype_dist), order='dist')
        
        self.num_grps = num_grps
        self.grp_sizes = []
        default_grp_size = math.floor(self.n / self.num_grps)
        last_grp_size = self.n % default_grp_size + default_grp_size
        for i in range(self.num_grps):
            if i == self.num_grps - 1:
                self.grp_sizes.append(last_grp_size)
            else:
                self.grp_sizes.append(default_grp_size)

    def run(self):
        ret = np.zeros((self.n,self.m))
        start = 0
        for i in range(self.num_grps):
            grp_size = self.grp_sizes[i]
            popularity_grp = self.popularity[start:start+grp_size]
            a=0
            for (idx, _) in popularity_grp:
                loc = self.distance[start+a]['idx']
                ret[int(idx)][loc] = 1
                a+=1
            start += grp_size
        return ret