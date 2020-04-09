import numpy as np
import random as rd
import math

class OrderGen:
    def __init__(self, num_skus, max_order_size):
        self.num_skus = num_skus
        self.max_order_size = max_order_size

    def generate(self, num_orders):
        '''generates a new set of orders, updating summary'''
        orders = []
        for i in range(num_orders):            
            order = []
            order_size = rd.randint(1,self.max_order_size)
            
            for j in range(order_size):
                sku = rd.randint(1,self.num_skus)
                order.append(sku)
            orders.append(order)
        self.orders = orders
        return orders

    def generate_exact(self, num_items):
        num_items_eighty = int(math.ceil(num_items*0.8))
        num_items_twenty = num_items - num_items_eighty
        
        first_num_skus = math.floor(self.num_skus * 0.2)
        if first_num_skus <= 1:
            first_num_skus += 1
        sku_sublist1 = list(range(1,1+first_num_skus))
        
        second_num_skus = self.num_skus - first_num_skus
        sku_sublist2 = list(range(1+first_num_skus, 1+first_num_skus+second_num_skus))
        order_eighty = self.aux(num_items_eighty, sku_sublist1)
        order_twenty = self.aux(num_items_twenty, sku_sublist2)
        ret = order_eighty + order_twenty
        ret = self.shuffle(ret)
        self.orders = ret
        return ret

    def shuffle(self, order_list):
        order_size_list = []
        flat_list = [item for l in order_list for item in l]
        rd.shuffle(flat_list)
        
        for order in order_list:
            order_size_list.append(len(order))
        new_order_list = []
        
        num_orders = len(order_list)
        start = 0
        for i in range(num_orders):
            order_size = order_size_list[i]
            new_order_list.append(flat_list[start:start+order_size])
            start += order_size

        return new_order_list            

    def aux(self, num_items, sku_list):
        '''generate the exact number of items specified by num_items, updating summary'''
        '''control 20% of SKU to appear in 80% of all orders'''
        orders = []
        size = len(sku_list)
        has_next_order = True
        num_left = num_items
        while has_next_order:
            order = []
            order_size = rd.randint(1,self.max_order_size)
            if order_size > num_left:
                order_size = num_left
                has_next_order = False
            for i in range(order_size):
                sku = sku_list[rd.randint(0,size-1)]
                order.append(sku)
            orders.append(order)
            num_left -= order_size
        return orders


    def save(self,dest):
        with open(dest,'w') as f:
            for order in self.orders:
                order_str = ','.join(map(str,order))
                f.write('%s\n' % order_str)
