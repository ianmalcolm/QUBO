def index_1_q_to_l_1(i,j,m):
    '''
    assumes a (n by m) matrix
    '''
    return (i-1)*m + j

def index_1_l_to_q_1(k,m):
    '''
    assumes a (n by m) decision matrix.
    linear index higher than (n*m) is ancillary.
    '''
    r = k % m
    j = r
    i = (k-r) / m + 1
    return (i,j)

def index_1_to_0(i):
    return i-1

def var_str(id, number):
    return id + str(number)
