import numpy as np
import math
import utils.index as idx

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
    
    def gen_F(self, is_for_items=True):
        '''generate F for all orders'''
        ret = np.zeros((self.num_SKUs+1, self.num_SKUs+1),dtype=np.int32)
        orders = self.order_str.splitlines()
        for order in orders:
            partial_F = self.gen_interaction_frequency(order)
            ret += partial_F
        #qty stores quantities of SKUs with 1-based index
        self.qty = ret[0]
        if not is_for_items:
            return ret
        else:
            num_items = int(sum(self.qty))
            _ret = np.zeros((num_items, num_items),dtype=np.int32)
            
            # maps 1-based item indices to 1-based sku indices
            sku_indices = np.zeros(num_items + 1)
            a=1
            for i in range(1,len(self.qty)):
                qty_i = self.qty[i]
                for j in range(int(qty_i)):
                    # i is the current sku index
                    sku_indices[a] = i
                    a+=1
            sku_indices = sku_indices.astype(int)

            print(self.qty)
            # construct F for items as upper-triangular
            for i in range(1,num_items+1):
                for j in range(1,num_items+1):
                    if i==j:
                        _ret[i-1][i-1] = self.qty[sku_indices[i]]
                    elif i<j:
                        _ret[i-1][j-1] = ret[sku_indices[i]][sku_indices[j]]
                    else:
                        pass
            return _ret
    
    def gen_interaction_frequency(self, order):
        '''generate partial F for a single order'''
        ret = np.zeros((self.num_SKUs+1, self.num_SKUs+1), dtype=np.int32)
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

    