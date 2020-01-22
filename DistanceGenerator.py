import numpy as np
import math

class DistanceGenerator:
    '''a priori generates distance matrix for some routing strategy'''
    def __init__(self, num_rows, num_cols, DIST_VERTICAL, DIST_HORIZONTAL, group_num_rows=None, group_num_cols=None):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_tot = num_cols * num_rows
        self.D = np.zeros((self.num_tot, self.num_tot))
        self.DIST_VERTICAL = DIST_VERTICAL
        self.DIST_HORIZONTAL = DIST_HORIZONTAL
        
        self.group_num_cols = group_num_cols
        self.group_num_rows = group_num_rows
    
    def gen_Euclidean(self):
        '''
            returns a symmetric D
        '''
        for y in range(self.num_rows):
            for x in range(self.num_cols):
                #(x,y)
                index = self.find_D_index(x,y,self.num_rows)
                dist = math.pow(x,2) + math.pow(y,2)
                self.D[index][index] = dist
                for y_prime in range(self.num_rows):
                    for x_prime in range(self.num_cols):
                        #(x',y'). dist = (y-y')^2 + (x-x')^2
                        dist = math.pow((y_prime-y),2) + math.pow((x_prime-x),2)
                        index_prime = self.find_D_index(x_prime,y_prime,self.num_rows)
                        self.D[index][index_prime] = dist
        return self.D
    
    def gen_Dprime(self,D):
        Dprime_length = self.group_num_rows * self.group_num_cols + 1
        Dprime = np.zeros((Dprime_length,Dprime_length))
        #consider each pair of locations
        for i in range(0,self.group_num_rows):
            for j in range(0,self.group_num_cols):
                Dprime_index = self.find_D_index(i,j,self.group_num_rows)
                D_index = self.find_D_index(i,j,self.num_rows)
                #fill up distance from depot
                Dprime[0][Dprime_index] = D[0][D_index]
                Dprime[Dprime_index][0] = D[D_index][0]
                #fill up distance with all other locations
                for k in range(0,self.group_num_rows):
                    for l in range(0,self.group_num_cols):
                        Dprime_index_other = self.find_D_index(k,l,self.group_num_rows)
                        D_index_other = self.find_D_index(k,l,self.num_rows)
                        Dprime[Dprime_index][Dprime_index_other] = D[D_index][D_index_other]
        return Dprime

    def find_D_index(self,i,j,num_rows):
        '''maps 0 based coordinates to 0-based index of D'''
        return (j*num_rows + i)

    def gen_S_shape(self):
        '''computes S-shaped routing distances. 
            assumed layout: 1,2,2,...for odd #columns
                            1,2,2,...1 for even #columns
            assumed routing strategy: up, down, up, down, ... until exhaustion

            returns:
                A symmetric D
        '''
        
        distance_from_depot = 0
        goingUp = True

        # store D indices
        list_even = []
        list_odd = []
        
        list_distances = []
        
        j=0
        hasNeighbor=True
        while j < self.num_cols:
            print("considering column ", j)
            
            if (self.num_cols % 2 and j==self.num_cols-1):
                hasNeighbor=False
            if not goingUp:
                starting_row = self.num_rows-1
                ending_row = -1
                step = -1
            else:
                starting_row = 0
                ending_row = self.num_rows
                step = 1
            
            i = starting_row
            while i != ending_row:
                print("considering item at", i, j)
                list_even.append(self.find_D_index(i,j,self.num_rows))
                if hasNeighbor:
                    print("considering neighboring item at", i, j+1)
                    list_odd.append(self.find_D_index(i,j+1,self.num_rows))
                else:
                    list_odd.append(-1)
                list_distances.append(distance_from_depot)
                
                distance_from_depot += self.DIST_VERTICAL
                if (goingUp and i==self.num_rows-1) or ((not goingUp) and i==0):
                    distance_from_depot += self.DIST_HORIZONTAL
                i += step
            j+=2
            goingUp = not goingUp
        
        for k in range(len(list_even)):
            print(k, list_even[k], list_distances[k])
            self.D[list_even[k]][list_even[k]] = list_distances[k]
            if list_odd[k] != -1:
                self.D[list_odd[k]][list_odd[k]] = list_distances[k]
            for l in range(k+1,len(list_even)):
                self.D[list_even[k]][list_even[l]] = list_distances[l]-list_distances[k]
                if list_odd[k] != -1:
                    self.D[list_odd[k]][list_even[l]] = list_distances[l]-list_distances[k]
                if list_odd[l] != -1:
                    self.D[list_even[k]][list_odd[l]] = list_distances[l]-list_distances[k]
                    if list_odd[k] != -1:
                        self.D[list_odd[k]][list_odd[l]] = list_distances[l]-list_distances[k]
        self.D = self.D + np.transpose(self.D)
        for i in range(self.D.shape[0]):
            self.D[i][i] = self.D[i][i] / 2
        return self.D