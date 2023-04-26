from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return 4/(1+x**2)

if __name__ == "__main__":
    n = 14
    trapez = np.empty(shape=(n,2))
    gauss = np.empty(shape=(n,2))
    for i in range(n):
        tol = 10**-i
        trapez[i] = integrate.quad_vec(f,0,1,epsrel=tol,quadrature='trapezoid')
        gauss[i] = integrate.quad_vec(f,0,1,epsrel=tol,quadrature='gk21')
    print(trapez[:,1])
    plt.plot(trapez[:,1])
    # print(gauss)
    # print(trapez)