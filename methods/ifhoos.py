import numpy as np
import itertools

class IFHOOS:
    def __init__(self, F, D):
        self.F = F.copy()
        self.D = D.copy()
        self.size = F.shape[0]

    def run(self):
        ret = np.zeros((self.size,self.size))

        pairs_list = []
        for i,j in itertools.product(range(self.size),range(self.size)):
            if i <= j:
                pairs_list.append((i,j), self.F[i][j])
        pairs_dtype = [('indices',(int,int)), ('f', int)]
        pairs = np.sort(np.array(pairs_list, dtype=pairs_dtype),order='f')[::-1]
        
        indices_item = np.arange(0,self.size)
        dtype_pop = [('idx',int),('pop',int)]
        self.popularity = np.sort(np.array(list(zip(indices_item,np.diag(self.F))),dtype=dtype_pop), order='pop')[::-1]

        indices_location = np.arange(0,self.size)
        dtype_dist = [('idx',int),('dist',int)]
        self.distance = np.sort(np.array(list(zip(indices_location,np.diag(self.D))),dtype=dtype_dist), order='dist')

        # a map from item index to allocated location
        item_allocated = np.full(self.size, fill_value=-1)
        # a map from location index to allocated item
        loc_allocated = np.full(self.size,fill_value=-1)

        ret = self.allocate_singles(ret, item_allocated, loc_allocated)
        ret = self.allocate_pairs(ret, pairs, self.distance, item_allocated, loc_allocated, 0.01)
        ret = self.sweep(ret, item_allocated, loc_allocated, self.popularity, self.distance)

        return ret

    def sweep(self, ret, item_allocated, loc_allocated, popularity, distance):
        items_unallocated = []
        locs_unallocated = []
        for i in range(len(item_allocated)):
            if item_allocated[i] == -1:
                items_unallocated.append(i)
        for i in range(len(loc_allocated)):
            if loc_allocated[i] == -1:
                locs_unallocated.append(i)
        
        for i in range(len(items_unallocated)):
            item = items_unallocated[i]
            coi_loc = self.get_coi_location(item)
            if loc_allocated[coi_loc] == -1:
                ret[item][coi_loc] = 1
                item_allocated[item] = coi_loc
                loc_allocated[coi_loc] = item
                items_unallocated.remove(item)
                locs_unallocated.remove(coi_loc)
        
        dtype_pop = [('idx',int),('pop',int)]
        dtype_dist = [('idx',int),('dist',int)]

        remainder_items = []
        remainder_locs = []
        for i in range(len(items_unallocated)):
            item = items_unallocated[i]
            remainder_items.append((item,(popularity[item])['pop']))
        for i in range(len(locs_unallocated)):
            loc = locs_unallocated[i]
            remainder_locs.append((loc,(distance[loc])['dist']))
        remainder_items = np.sort(np.array(remainder_items, dtype=dtype_pop), order='pop')[::-1]
        remainder_locs = np.sort(np.array(remainder_locs, dtype=dtype_dist), order='dist')

        for i in range(remainder_items):
            item = remainder_items[i]
            loc = remainder_locs[i]
            ret[item][loc] = 1

        return ret

    def get_coi_location(self, index):
        rank = np.where(self.popularity==index)[0]
        coi_pos = (self.distance[rank])['idx']
        return coi_pos

    def allocate_pairs(self, ret, pairs, distance, item_allocated, loc_allocated, beta):
        def prepare_loc_set(index):
            loc_set = set()
            d = distance[index]['dist']
            d_minus = d*(1-beta)
            d_plus = d*(1+beta)
            
            curr = index
            while d_minus<=distance[curr]['dist']:
                if loc_allocated[index] == -1:
                    loc_set.add(curr)
                curr = curr - 1
            
            curr = index
            while distance[curr]['dist']<=d_plus:
                if loc_allocated[index] == -1:
                    loc_set.add(curr)
                curr = curr + 1
            return loc_set

        pair_size = len(pairs)
        for i in range(pair_size):
            ((j1,j2),_) = pairs[i]
            loc_j1_final = item_allocated[j1]
            loc_j2_final = item_allocated[j2]
            
            if item_allocated[j1]==-1 and item_allocated[j2]==-1:
                locs_j1 = prepare_loc_set(j1)
                locs_j2 = prepare_loc_set(j2)
                
                min_dist = np.inf
                for loc_j1, loc_j2 in itertools.product(locs_j1, locs_j2):
                    dist = self.D[loc_j1][loc_j2] 
                    if dist <= min_dist:
                        min_dist = dist
                        loc_j1_final = loc_j1
                        loc_j2_final = loc_j2

            elif item_allocated[j1] and item_allocated[j2]:
                pass
            elif item_allocated[j1] and item_allocated[j2]==-1:
                locs_j2 = prepare_loc_set(j2)
                min_dist = np.inf
                for loc_j2 in locs_j2:
                    dist = self.D[loc_j1_final][loc_j2]
                    if dist <= min_dist:
                        min_dist = dist
                        loc_j2_final = loc_j2
                
            elif item_allocated[j1]==-1 and item_allocated[j2]:
                locs_j1 = prepare_loc_set(j1)
                for loc_j1 in locs_j1:
                    dist = self.D[loc_j2_final][loc_j1]
                    if dist <= min_dist:
                        min_dist = dist
                        loc_j1_final = loc_j1

            ret[j1][loc_j1_final] = ret[j2][loc_j2_final] = 1
            loc_allocated[loc_j1_final] = j1
            loc_allocated[loc_j2_final] = j2
            item_allocated[j1] = loc_j1_final
            item_allocated[j2] = loc_j2_final
            
        return ret

    def allocate_singles(self, ret, item_allocated, loc_allocated):
        singles = []
        for i in range(self.size):
            if sum(self.F[i])==0:
                singles.append(i)
        num_singles = len(singles)
        for i in range(num_singles):
            index = np.where(self.popularity==singles[i])[0]
            coi_pos = self.get_coi_location(singles[i])
            ret[index][coi_pos] = 1
            item_allocated[index] = coi_pos
            loc_allocated[coi_pos] = index
        return ret