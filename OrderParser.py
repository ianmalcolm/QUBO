import numpy as np
import math

class OrderParser:
    def __init__(self, order, num_SKUs):
        self.num_SKUs = num_SKUs
        self.order = order
        self.ret = np.zeros((num_SKUs+1, num_SKUs+1))
    
    @staticmethod
    def nCr(n,r):
        return math.factorial(n) / (math.factorial(r) * math.factorial(n-r))

    # generate partial F for a single order
    def gen_interaction_frequency(self):
        lines = self.order.splitlines()
        sku_types = set()
        for line in lines:
            sku = line.split(",")
            sku_type = int(sku[0])
            sku_types.add(sku_type)
            sku_qty = int(sku[1])
            self.ret[0][sku_type] = sku_qty

        sku_types_ls = list(sku_types)
        sku_types_ls.sort()
        for i in range(len(sku_types_ls)):
            for j in range(i+1,len(sku_types_ls)):
                x,y = (sku_types_ls[i], sku_types_ls[j])
                print(x,y)
                self.ret[x][y] = self.ret[0][x] * self.ret[0][y]
        self.ret = self.ret + self.ret.transpose()
        for i in range(self.num_SKUs):
            if self.ret[0][i+1] >1:
                self.ret[i+1][i+1] = self.nCr(self.ret[0][i+1],2)

        return self.ret