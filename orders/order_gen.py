import numpy as np
import random as rd

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
        '''generate the exact number of items specified by num_items, updating summary'''
        orders = []
        has_next_order = True
        num_left = num_items
        while has_next_order:
            order = []
            order_size = rd.randint(1,self.max_order_size)
            if order_size > num_left:
                order_size = num_left
                has_next_order = False
            for i in range(order_size):
                sku = rd.randint(1,self.num_skus)
                order.append(sku)
            orders.append(order)
            num_left -= order_size
        self.orders = orders
        return orders
    
    def save(self,dest):
        with open(dest,'w') as f:
            for order in self.orders:
                order_str = ','.join(map(str,order))
                f.write('%s\n' % order_str)
