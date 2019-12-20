def inspect_entries(F):
    size = F.shape[0]
    zeros = 0
    nonzeros = 0
    
    for i in range(size):
        for j in range(size):
            if F[i][j] == 0:
                zeros += 1
            else:
                nonzeros += 1
    
    return (zeros, nonzeros, zeros / nonzeros)

def inspect_upper(F):
    size = F.shape[0]
    zeros = 0
    nonzeros = 0
    
    for i in range(size):
        for j in range(size):
            if i<=j:
                if F[i][j] == 0:
                    zeros += 1
                else:
                    nonzeros += 1
    
    return (zeros, nonzeros, zeros / nonzeros)