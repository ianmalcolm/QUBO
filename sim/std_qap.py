import itertools

def obj(F,D,perm):
    energy = 0
    n = len(perm)
    for i,j in itertools.product(range(n),range(n)):
        k=perm[i]
        l=perm[j]
        energy += F[i][j]*D[k][l]
    return energy