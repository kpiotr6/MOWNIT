import numpy as np
def f1(x,j):
    return x**j
def f2(x,j):
    return (x-1900)**j
def f3(x,j):
    return (x-1940)**j
def f4(x,j):
    return ((x-1940)/40)**j

def vandermond(f,xs):
    n = len(xs)
    V = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            V[i][j] = f(xs[i],j)
    return V



if __name__ == '__main__':
    years = [1900,1910,1920,1930,1940,1950,1960,1970,1980]
    pop = [76212168,92228496,106021537,123202624,132164569,151325798,179323175,203302031,226542199]
    v = vandermond(f1,years)
    print(v)
