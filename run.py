from simanneal import Annealer
from orders.order_parser import OrderParser
from DistanceGenerator import DistanceGenerator
from problems.grouping import GroupingProblem
from problems.permutation import PermutationProblem
import os
import math
import numpy as np
import xlsxwriter
from methods.exterior_penalty import ExteriorPenaltyMethod
from problems.bunching import BunchingQAP
from ports.dwave import Dwave
from ports.classical_simanneal import ClassicalNeal

import utils.mtx as mtx

NUM_SKUS = 100
WAREHOUSE_NUM_COLS = 40
WAREHOUSE_NUM_ROWS = 20
DIST_VERTICAL = 1
DIST_HORIZONTAL = 1
ORDER_DIRNAME = 'orders'
F_FIRST_FILENAME = 'F.dat'
F_SECOND_FILENAME = 'F2.dat'
D_FILENAME = 'D.dat'
DPRIME_FILENAME = 'Dprime.dat'
QTY_FILENAME = 'qty.dat'
GROUPING_FILENAME = 'grouping.txt'
PERM_FILENAME = 'permutation'
BIGF_FILENAME = 'bigF.dat'
BIGD_FILENAME = 'bigD.dat'
BIGQTY_FILENAME = 'bigQty.dat'

group_num_cols = 20
group_num_rows = 20

def save_array(fname, arrname, arr, prefix=None):
    with open(fname,'w') as f:
        if prefix:
            f.write(prefix)
            f.write('\n')
        arr=arr.astype('int32')
        str = np.array2string(arr,separator=',')
        str = arrname+'=' + str + ';'
        f.write(str)

def main():
    order_parser = OrderParser("orders/order.txt", NUM_SKUS, threshold=0)
    # F: (n by n) upper triangular interaction frequency matrix
    F = order_parser.gen_F()
    print(F)

    qty = order_parser.summary()

    problem = BunchingQAP(800,800,2,F)

    solver = ClassicalNeal()
    method = ExteriorPenaltyMethod(problem, solver)
    solution = method.run()
    input()
    
    #calculate D, m+1 by m+1
    D_gen = DistanceGenerator(
        WAREHOUSE_NUM_ROWS, 
        WAREHOUSE_NUM_COLS, 
        DIST_VERTICAL, 
        DIST_HORIZONTAL,
        group_num_rows, 
        group_num_cols
        )
    D = D_gen.gen_S_shape()
    print("D: ", D)


    #extract Dprime, xy+1 by xy+1, distance matrix for the first few locations
    Dprime = D_gen.gen_Dprime(D)
    print("Dprime: ", Dprime)

    x = np.zeros((WAREHOUSE_NUM_ROWS*WAREHOUSE_NUM_COLS,))

    save_array(F_FIRST_FILENAME,'F',F)
    save_array(F_SECOND_FILENAME,'F',F, prefix="NUM_SKUS="+str(NUM_SKUS)+';')
    save_array(D_FILENAME,'D',D)

    group_size = group_num_cols*group_num_rows
    save_array(DPRIME_FILENAME,'D',Dprime, prefix="NUM_LOCS="+str(group_size)+';')
    save_array(QTY_FILENAME,'qty',F[0])

    problem1 = GroupingProblem(group_size, NUM_SKUS, F)
    grouping = np.array(problem1.solve()).astype('int32')
    
    input()

    ######################
    #second stage
    ######################

    def find_sku_index(i):
        '''returns the 1-based sku index for a given item'''
        # NOTE: item index is 0 based
        ret = i+1
        for j in range(1,F.shape[1]):
            ret -= F[0][j]
            if ret <= 0:
                return j
        raise ValueError("bad item index that exceeds sku limit")

    num_items = sum(F[0])
    num_groups = math.ceil(num_items / group_size)
    for i in range(num_groups):
        qty = np.zeros(F.shape[1])
        item_list = grouping[:,i]
        print(item_list)
        for j in range(len(item_list)):
            if item_list[j]:
                sku = find_sku_index(j)
                print('item' + str(j) + 'belongs to sku' + str(sku))
                qty[sku] += item_list[j]
        print('qty'+str(i)+':', qty)
        save_array('qty_group'+str(i)+'.dat','qty',qty)

        problem2 = PermutationProblem(group_size,F,Dprime,qty,1)
        problem2.solve()
    
    '''
    #########################
    third stage process
    #########################
    '''
    # generate aggregate quantity
    bigQty = np.ones((num_groups+1,))
    bigQty[0]=0
    save_array(BIGQTY_FILENAME,'qty',bigQty)

    # generate aggregate D
    '''NOTE: 
        1) a group can cross column boundaries, so general Euclidean distance is calculated
        2) group locs are indexed in the same way as SKU locs are
        3) PRECONDITION: must divide!
    '''
    num_groups_x = int(WAREHOUSE_NUM_COLS / group_num_cols)
    num_groups_y = int(WAREHOUSE_NUM_ROWS / group_num_rows)
    bigD_gen = DistanceGenerator(num_groups_y,num_groups_x,1,1)
    bigD = bigD_gen.gen_Euclidean()
    save_array(BIGD_FILENAME, 'D', bigD, prefix="NUM_LOCS="+str(num_groups)+';')

    # generate aggregate F (num_groups+1 by num_groups+1)
    '''NOTE: index of a group follows reading order, but is 1-based.'''
    perms = []
    bigF = np.zeros((num_groups+1,num_groups+1))
    for i in range(num_groups):
        perm = np.loadtxt(PERM_FILENAME+str(i)+'.txt')
        perm = perm.astype('int32')
        perms.append(perm)
    #compute absolute popularity of each group
    for i in range(1,num_groups+1):
        bigF[0][i] = sum([F[0][j] for j in perms[i-1]])
        
    #compute interaction between each distinct pair of groups
    '''NOTE: a group is unique and does not interact with itself.'''
    for i in range(1,num_groups+1):
        for j in range(i+1,num_groups+1):
            p1 = perms[i-1]
            p2 = perms[j-1]
            # two groups interact as often as their individual elements interact
            for sku1 in p1:
                for sku2 in p2:
                    bigF[i][j] += F[sku1][sku2]
    bigF = bigF + bigF.transpose()
    save_array(BIGF_FILENAME,'F',bigF, prefix='NUM_SKUS='+str(num_groups)+';')          

if __name__ == "__main__":
    main()