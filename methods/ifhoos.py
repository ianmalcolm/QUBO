import numpy as np
import itertools
import utils.mtx as mtx

class IFHOOS:
    def __init__(self, F, D, beta):
        self.F = F.copy()
        self.D = D.copy()
        self.size = F.shape[0]
        self.beta = beta

    def run(self):
        ret = np.zeros((self.size,self.size))

        pairs_list = []
        for i,j in itertools.product(range(self.size),range(self.size)):
            if i < j:
                pairs_list.append(((i,j), self.F[i][j]))
        pairs_dtype = [('indices',int,(2,)), ('f', int)]
        pairs = np.sort(np.array(pairs_list, dtype=pairs_dtype),order='f')[::-1]

        #np.set_printoptions(threshold=np.inf)
        #print(pairs)
        #input()
        
        indices_item = np.arange(0,self.size)
        dtype_pop = [('idx',int),('pop',int)]
        self.popularity = np.sort(np.array(list(zip(indices_item,np.diag(self.F))),dtype=dtype_pop), order='pop')[::-1]


        indices_location = np.arange(0,self.size)
        dtype_dist = [('idx',int),('dist',int)]
        self.distance = np.sort(np.array(list(zip(indices_location,np.diag(self.D))),dtype=dtype_dist), order='dist')

        # a map from item index to allocated location
        item_allocated = np.full(self.size, fill_value=None)
        # a map from location index to allocated item
        loc_allocated = np.full(self.size,fill_value=None)

        np.set_printoptions(threshold=np.inf)
        ret = self.allocate_singles(ret, item_allocated, loc_allocated)
        # print("after singles: ", item_allocated)
        # print(ret)
        ret = self.allocate_pairs(ret, pairs, self.distance, item_allocated, loc_allocated, beta=self.beta)
        print("after pairs: ", item_allocated, loc_allocated)
        print(ret)
        ret = self.sweep(ret, item_allocated, loc_allocated, self.popularity, self.distance)
        # print("Finally: ", item_allocated)
        # print(ret)
        return ret

    def sweep(self, ret, item_allocated, loc_allocated, popularity, distance):
        items_unallocated = []
        locs_unallocated = []
        for i in range(len(item_allocated)):
            if item_allocated[i] is None:
                items_unallocated.append(i)
        for i in range(len(loc_allocated)):
            if loc_allocated[i] is None:
                locs_unallocated.append(i)
        # print(items_unallocated)
        # print(locs_unallocated)
        # input()
        for i in range(len(items_unallocated)):
            item = items_unallocated[i]
            coi_loc = self.get_coi_location(item)
            if loc_allocated[coi_loc] is None:
                ret[item][coi_loc] = 1
                item_allocated[item] = coi_loc
                loc_allocated[coi_loc] = item
                items_unallocated.remove(item)
                locs_unallocated.remove(coi_loc)
        
        # print(items_unallocated)
        # print(locs_unallocated)
        # input()
        dtype_pop = [('idx',int),('pop',int)]
        dtype_dist = [('idx',int),('dist',int)]

        remainder_items = []
        remainder_locs = []
        for i in range(len(items_unallocated)):
            item = items_unallocated[i]
            remainder_items.append((item,self.F[item][item]))
        for i in range(len(locs_unallocated)):
            loc = locs_unallocated[i]
            remainder_locs.append((loc,self.D[loc][loc]))
        remainder_items = np.sort(np.array(remainder_items, dtype=dtype_pop), order='pop')[::-1]
        remainder_locs = np.sort(np.array(remainder_locs, dtype=dtype_dist), order='dist')

        for i in range(len(remainder_items)):
            item = remainder_items[i]['idx']
            loc = remainder_locs[i]['idx']
            print(item,loc)
            ret[item][loc] = 1
            item_allocated[item] = loc
            loc_allocated[loc] = item

        return ret

    def get_coi_location(self, index):
        for i in range(self.size):
            if self.popularity[i]['idx'] == index:
                return self.distance[i]['idx']
    
    def get_coi_location_index(self, item):
        for i in range(self.size):
            if self.popularity[i]['idx'] == item:
                return i

    def find_closest_locs_diff(self, set1, set2):
        min_dist = np.inf
        min_loc1 = None
        min_loc2 = None
        for loc1, loc2 in itertools.product(set1,set2):
            if not loc1==loc2:
                dist = self.D[loc1][loc2]
                if dist <= min_dist:
                    min_dist = dist
                    min_loc1 = loc1
                    min_loc2 = loc2
        return min_loc1, min_loc2
    
    def find_nearest_loc(self, reference_loc, locset):
        min_dist = np.inf
        loc_final = None
        for loc in locset:
            dist = self.D[reference_loc][loc]
            if dist <= min_dist:
                min_dist = dist
                loc_final = loc
        return loc_final

    def allocate_pairs(self, ret, pairs, distance, item_allocated, loc_allocated, beta):
        np.set_printoptions(threshold=np.inf)
        def update_records(j1,j2,loc_j1,loc_j2):
            if loc_j1==63:
                print("loc of 63: ",j1)
            if loc_j2==63:
                print("loc of 63: ",j2)
            if j1 == 32:
                print("j1 of 32:",loc_j1)
            if j2 == 32:
                print("j2 of 32:",loc_j2)
            item_allocated[j1] = loc_j1
            item_allocated[j2] = loc_j2
            if not loc_j1 is None:
                loc_allocated[loc_j1] = j1
                ret[j1][loc_j1] = 1
            if not loc_j2 is None:
                loc_allocated[loc_j2] = j2
                ret[j2][loc_j2] = 1
            return
        
        def prepare_loc_set(index):
            loc_set = set()
            coi_location = self.get_coi_location(index)
            coi_location_index = self.get_coi_location_index(index)
            
            d = distance[coi_location_index]['dist']
            d_minus = d*(1-beta)
            d_plus = d*(1+beta)
            
            curr_index = coi_location_index
            curr_location = coi_location
            count = 0
            while curr_index >= 0 and d_minus<=distance[curr_index]['dist']:
                curr_location = distance[curr_index]['idx']
                if loc_allocated[curr_location] is None:
                    loc_set.add(curr_location)
                curr_index -= 1
                count += 1
            
            curr_index = coi_location_index
            curr_location = coi_location
            while curr_index < self.size and distance[curr_index]['dist']<=d_plus:
                curr_location = distance[curr_index]['idx']
                if loc_allocated[curr_location] is None:
                    loc_set.add(curr_location)
                curr_index += 1

            return loc_set

        pair_size = len(pairs)
        for i in range(pair_size):
            ((j1,j2),_) = pairs[i]
            loc_j1_final = item_allocated[j1]
            loc_j2_final = item_allocated[j2]
            
            if item_allocated[j1] is None and item_allocated[j2] is None:
                locs_j1 = prepare_loc_set(j1)
                locs_j2 = prepare_loc_set(j2)
                
                loc_j1_final, loc_j2_final = self.find_closest_locs_diff(locs_j1,locs_j2)
                update_records(j1,j2,loc_j1_final,loc_j2_final)

            elif (not item_allocated[j1] is None) and (not item_allocated[j2] is None):
                pass

            elif (not item_allocated[j1] is None) and item_allocated[j2] is None:
                locs_j2 = prepare_loc_set(j2)
                loc_j2_final = self.find_nearest_loc(loc_j1_final, locs_j2)
                update_records(j1,j2,loc_j1_final,loc_j2_final)
                
            elif (item_allocated[j1] is None) and (not item_allocated[j2] is None):
                locs_j1 = prepare_loc_set(j1)
                loc_j1_final = self.find_nearest_loc(loc_j2_final, locs_j1)
                update_records(j1,j2,loc_j1_final,loc_j2_final)
            
        print(item_allocated)
        print(loc_allocated)
        return ret

    def allocate_singles(self, ret, item_allocated, loc_allocated):
        singles = []
        for i in range(self.size):
            if sum(self.F[i])==self.F[i][i]:
                singles.append(i)
        num_singles = len(singles)
        for i in range(num_singles):
            coi_location = self.get_coi_location(singles[i])
            ret[singles[i]][coi_location] = 1
            item_allocated[singles[i]] = coi_location
            loc_allocated[coi_location] = singles[i]
        
        return ret