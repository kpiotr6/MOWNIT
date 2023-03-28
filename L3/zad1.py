import numpy as np
from random import randint
import matplotlib.pyplot as plt

def f1(x,j):
    return x**j
def f2(x,j):
    return (x-1900)**j
def f3(x,j):
    return (x-1940)**j
def f4(x,j):
    return ((x-1940)/40)**j

def f2_val(x):
    return (x-1900)
def f3_val(x):
    return (x-1940)
def f4_val(x):
    return (x-1940)/40

def vandermond(f,xs):
    n = len(xs)
    V = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            V[i][j] = f(xs[i], j)
    return V

def horner(x,c):
    result = 0 
    for i in range(len(c)-1,-1,-1):
        result = (result*x) + c[i]
    return result

def LagrangeInterpolation(xs, ys, x):
    sol = 0
    t = 0

    for i in range(0, len(xs)):
        t = 1
        for j in range(0, len(ys)):
            if(j != i):
                t = t*((x-xs[j])/(xs[i]-xs[j]))
        sol += t*ys[i]

    return sol

def _poly_newton_coefficient(x, y):
    """
    x: list or np array contanining x data points
    y: list or np array contanining y data points
    """

    m = len(x)

    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])

    return a

def newton_polynomial(x_data, y_data, x):
    """
    x_data: data points at x
    y_data: data points at y
    x: evaluation point(s)
    """
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1  # Degree of polynomial
    p = a[n]

    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p

    return p


if __name__ == '__main__':
    years = np.array([1900,1910,1920,1930,1940,1950,1960,1970,1980])
    pop = np.array([76212168,92228496,106021537,123202624,132164569,151325798,179323175,203302031,226542199])
    v1 = vandermond(f1,years)
    v2 = vandermond(f2,years)
    v3 = vandermond(f3,years)
    v4 = vandermond(f4,years)

    cond1 = np.linalg.cond(v1)
    cond2 = np.linalg.cond(v2)
    cond3 = np.linalg.cond(v3)
    cond4 = np.linalg.cond(v4)
    
    best_cond = min(cond1, cond2, cond3, cond4)

    print("cond1 = ", cond1, "\ncond2 = ", cond2, "\ncond3 = ", cond3, "\ncond4 = ", cond4)
    
    c = np.dot(np.linalg.inv(v4), pop)
    print(c)
    
    x_vals = np.arange(1900, 1981, 1)
    y_vals = np.zeros(81)
    for i in range(len(x_vals)):
        y_vals[i] = horner(f4_val(x_vals[i]), c)
    
    plt.plot(x_vals, y_vals)
    plt.show()

    # print(horner(f4_val(1980), c))
    # print(calc_polynom_naive(ct, 1920))
    # print(LagrangeInterpolation(years, pop, 1975))
    # print(newton_polynomial(years,pop,1970))
    # print(horner(1900, newton_poly(divided_diff(years,pop),years,1900)))

    
    # http://www.algorytm.org/procedury-numeryczne/interpolacja-lagrange-a/inter-lagrange-j.html
