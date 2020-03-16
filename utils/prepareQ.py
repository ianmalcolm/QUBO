import numpy as np
import utils.index as idx

def computeQ(F,D):
    '''compute the essential matrix Q'''
    # take f times d
    _F = F.reshape(-1,1)
    _D = D.reshape(1,-1)
    n = F.shape[0]
    m = D.shape[0]

    _Q = np.matmul(_F, _D)

    Q = np.zeros((n*m, n*m),dtype=np.int16)
    for i in range(n):
        for j in range(n):
            a = i*n + j
            submatrix = _Q[a,:]
            submatrix = submatrix.reshape(m,m)
            upper_left_x, upper_left_y = (i*m, j*m)
            Q[upper_left_x:upper_left_x+m,upper_left_y:upper_left_y+m] += submatrix.astype(np.int16)
    
    return Q

def initialise_flow_matrix(F,D):
    n = F.shape[0]
    m = D.shape[0]
    ret = np.zeros((m*n, m*n),dtype=np.int32)
    for i in range(1,n+1):
        for j in range(1,n+1):
            for k in range(1,m+1):
                for l in range(1,m+1):
                    # X is n by m
                    x_ik = idx.index_1_q_to_l_1(i,k,m)
                    x_jl = idx.index_1_q_to_l_1(j,l,m)
                    if x_ik == x_jl:
                        ret[x_ik-1][x_jl-1] = F[i-1][j-1] * D[k-1][k-1]
                    elif x_ik < x_jl:
                        ret[x_ik-1][x_jl-1] = F[i-1][j-1] * D[k-1][l-1]
        
    return ret

# F = np.array([[1,2,3],[0,1,5],[0,0,2]])
# D = np.array([[2,1,4],[0,3,2],[0,0,6]])
# computeQ(F,D)
# initialise_flow_matrix(F,D)