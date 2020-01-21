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
            threshold

        outputs:
            interaction frequency matrix (f_ij) of size (num_skus+1 * num_skus+1)
    '''
    def __init__(self, file, num_SKUs, threshold):
        self.num_SKUs = num_SKUs
        with open(file, 'r') as f:
            self.order_str = f.read()
        self.qty = np.zeros(num_SKUs+1)
        self.threshold = threshold
    
    @staticmethod
    def nCr(n,r):
        return math.factorial(n) / (math.factorial(r) * math.factorial(n-r))

    def summary(self):
        '''returns the qty array'''
        return self.qty
    
    def gen_F(self, is_for_items=True):
        '''generate F for all orders
        
            returns: if is_for_items then symmetric F (n by n)
                    else F(num_SKUs by num_SKUs).
                    Everything 0 based.
        '''
        
        old_F = np.zeros((self.num_SKUs+1, self.num_SKUs+1),dtype=np.int32)
        orders = self.order_str.splitlines()
        for order in orders:
            old_F += self.gen_interaction_frequency(order)

        #qty stores quantities of SKUs with 1-based index
        self.qty = old_F[0]
        if not is_for_items:
            return old_F[1:, 1:]
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

            print("quantities of SKUs: ")
            print(self.qty)
            # construct F for items as fully symmetric
            for i in range(1,num_items+1):
                for j in range(1,num_items+1):
                    if i==j:
                        # interaction between 2 identical items = its popularity
                        _ret[i-1][i-1] = self.qty[sku_indices[i]]
                    else:
                        # NOTE: when sku_indices[i] == sku_indices[j], 
                        #       the value is on diagonal of old_F which is the nC2 definition.
                        if old_F[sku_indices[i]][sku_indices[j]] > self.threshold:
                            _ret[i-1][j-1] = old_F[sku_indices[i]][sku_indices[j]]
            print("Flow matrix: ")
            #np.set_printoptions(threshold=np.inf)
            print(_ret)
            np.savetxt("Flow.txt",_ret, fmt='%d')
            np.set_printoptions(threshold=6)
            return _ret
    
    def gen_interaction_frequency(self, order):
        '''generate partial F for a single order
        
            returns: F (n+1 by n+1). 
                    F is Symmetric. 
                    Zero-th row represents quantities of SKUs.
                    Diagonal represents nC2 of a certain SKU appearing in a single order.
        '''
        ret = np.zeros((self.num_SKUs+1, self.num_SKUs+1), dtype=np.int32)
        sku_quantities = {}
        items = order.split(",")
        for item in items:
            if not item in sku_quantities:
                sku_quantities[item] = 1
            else:
                sku_quantities[item] += 1
        #print(sku_quantities)
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

        # count interaction frequency between two different items
        # if frequency does not exceed threshold value, treat it as 0
        sku_types_ls.sort()
        for i in range(len(sku_types_ls)):
            for j in range(i+1,len(sku_types_ls)):
                x,y = (sku_types_ls[i], sku_types_ls[j])
                #print(x,y)
                freq = sku_quantities[str(x)] * sku_quantities[str(y)]
                ret[x][y] = freq
        ret = ret + ret.transpose()
        
        # compute F[i][i]
        # interaction of a SKU with itself happens when 2 or more of the same SKU appear in a single order.
        # it is defined as nC2 which agrees with that between different SKUs
        for i in range(len(sku_types_ls)):
            sku = sku_types_ls[i]
            if sku_quantities[str(sku)] >1:
                ret[sku][sku] = self.nCr(sku_quantities[str(sku)],2)

        return ret

    