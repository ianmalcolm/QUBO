import numpy as np
import math

class RouteEvaluator:
    def __init__(self, qty, order_set, num_cols, num_rows, dist_ver, dist_hor, n,m):
        self.order_set = [[int(sku_str) for sku_str in order] for order in order_set]
        self.qty = qty
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_skus = len(self.qty)
        self.dist_ver = dist_ver
        self.dist_hor = dist_hor
        self.n = n
        self.m = m
    
        # itos - item to sku
        # ltoi - location to item
        # stol - sku to location(s)
        self.itos_dict = {}
        a = 0
        for i in range(self.num_skus):
            for j in range(self.qty[i]):
                self.itos_dict[a] = i
                a += 1
    
    def item_to_sku(self, item_index):
        return self.itos_dict[item_index]    

    def make_map(self, solution_mtx):
        ltoi = {}
        stol = {}
        for i in range(self.n):
            for k in range(self.m):
                if solution_mtx[i][k]:
                    ltoi[k] = self.item_to_sku(i)
                    if self.item_to_sku(i) not in stol:
                        stol[self.item_to_sku(i)] = [k]
                    else:
                        stol[self.item_to_sku(i)].append(k)
        print(ltoi)
        print(stol)
        return ltoi, stol

    def make_c(self):
        c = np.zeros(self.m)
        columns = int(math.ceil(self.m / (self.num_rows * 2)))
        for i in range(columns):
            for x in
                for y in :
                    c[] = i
            for j in range(self.num_rows * self.num_cols):
                c[j] = i
        return c

    def run(self, solution_mtx):
        distance = 0
        
        ltoc = self.make_c()
            
        def locate(map, sku):
            return min(map[sku])
        
        ltoi, stol = self.make_map(solution_mtx)
        
        num_orders = len(self.order_set)
        for o in range(num_orders):
            order = self.order_set[o]
            locs = []
            for i in order:
                # i is SKU number. Must minus 1 to get 0-based number.
                loc = locate(stol, i-1)
                locs.append(loc)
                ltoi[loc] = -1
                stol[i-1].remove(loc)
            locs = np.sort(locs)
            print(locs)
            cols = set()
            for l in locs:
                cols.add(ltoc[l])
            cols = np.sort(list(cols))
            print(cols)
            curr_c = 0
            for c in cols:
                distance += self.dist_hor * (c-curr_c)
                distance += self.num_rows * self.dist_ver
                curr_c = c
            if len(cols) % 2:
                distance += self.num_rows * self.dist_ver
            distance += curr_c * self.dist_hor

        return distance