import numpy as np
import math

class OrderParser:
    '''Parses a string of a set of orders in the following format:
        orders := {order\n}+
        order := SKU_NUM , order | epsilon
        Intuitively, an order is a line of comma separated integers

        Then computes interaction frequency

        parameters:
            num_skus
            file

        outputs:
            interaction frequency matrix (f_ij) of size (num_skus+1 * num_skus+1)
    '''
    def __init__(self, file, num_SKUs):
        self.num_SKUs = num_SKUs
        with open(file, 'r') as f:
            self.order_str = f.read()
        self.qty = np.zeros(num_SKUs+1)
    
    @staticmethod
    def nCr(n,r):
        return math.factorial(n) / (math.factorial(r) * math.factorial(n-r))

    def summary(self):
        '''returns the qty array'''
        return self.qty
        
    def gen_F(self):
        '''generate F for all orders'''
        ret = np.zeros((self.num_SKUs+1, self.num_SKUs+1))
        orders = self.order_str.splitlines()
        for order in orders:
            partial_F = self.gen_interaction_frequency(order)
            ret += partial_F
        self.qty = ret[0]
        return ret
    
    def gen_interaction_frequency(self, order):
        '''generate partial F for a single order'''
        ret = np.zeros((self.num_SKUs+1, self.num_SKUs+1))
        sku_quantities = {}
        items = order.split(",")
        for item in items:
            if not item in sku_quantities:
                sku_quantities[item] = 1
            else:
                sku_quantities[item] += 1
        print(sku_quantities)
        for (sku_type, sku_quantity) in sku_quantities.items():
            try:
                ret[0][int(sku_type)] = sku_quantity
            except TypeError:
                raise AssertionError("sku_type cannot be non-integer")

        sku_types_ls = list(sku_quantities.keys())
        try:
            sku_types_ls = [int(t) for t in sku_types_ls]
        except TypeError:
            raise AssertionError("sku_type cannot be non-integer")

        sku_types_ls.sort()
        for i in range(len(sku_types_ls)):
            for j in range(i+1,len(sku_types_ls)):
                x,y = (sku_types_ls[i], sku_types_ls[j])
                print(x,y)
                ret[x][y] = sku_quantities[str(x)] * sku_quantities[str(y)]
        ret = ret + ret.transpose()
        
        for i in range(len(sku_types_ls)):
            sku = sku_types_ls[i]
            if sku_quantities[str(sku)] >1:
                ret[sku][sku] = self.nCr(sku_quantities[str(sku)],2)

        return ret

    