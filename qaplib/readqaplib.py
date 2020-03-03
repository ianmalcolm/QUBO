import numpy as np
def readqaplib(filename):
    raw_data = []
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            numbers = line.split()
            for i in numbers:
                raw_data.append(int(i))
    
    size = raw_data[0]
    data = raw_data[1:]

    half_point = size * size
    mtx1 = np.array(data[:half_point],dtype=int).reshape((size,size))
    mtx2 = np.array(data[half_point:],dtype=int).reshape((size,size))
    np.set_printoptions(threshold=np.inf)
    #print(mtx1)
    #print(mtx2)
    return (mtx1,mtx2)