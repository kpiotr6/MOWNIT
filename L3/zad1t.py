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

def lagrange_polynomial(x_values, y_values, x):
    sol = 0
    t = 0
    for i in range(0, len(x_values)):
        t = 1
        for j in range(0, len(y_values)):
            if(j != i):
                t = t*((x-x_values[j])/(x_values[i]-x_values[j]))
        sol += t*y_values[i]

    return sol

def coefficients(x, y):
    """
    x: list or np array contanining x data points
    y: list or np array contanining y data points
    """
    m = len(x)
    x = np.array(x,dtype=float)
    a = np.array(y,dtype=float)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
    return a


def newton_polynomial(c,x_values, x):
    """
    x_data: data points at x
    y_data: data points at y
    x: evaluation point(s)
    """
    n = len(x_values) - 1  # Degree of polynomial
    result = 0
    for i in range(len(x_values)-1,-1,-1):
        result = result*(x-x_values[i]) + c[i]
    return result


if __name__ == '__main__':
    years = np.array([1900,1910,1920,1930,1940,1950,1960,1970,1980])
    pop = np.array([76212168,92228496,106021537,123202624,132164569,151325798,179323175,203302031,226542199])
    pop_round = np.round(pop,-6)
    # a)
    v1 = vandermond(f1,years)
    v2 = vandermond(f2,years)
    v3 = vandermond(f3,years)
    v4 = vandermond(f4,years)
    print(v4)
    # b)
    cond1 = np.linalg.cond(v1)
    cond2 = np.linalg.cond(v2)
    cond3 = np.linalg.cond(v3)
    cond4 = np.linalg.cond(v4)
    
    best_cond = min(cond1, cond2, cond3, cond4)

    print("cond1 = ", cond1, "\ncond2 = ", cond2, "\ncond3 = ", cond3, "\ncond4 = ", cond4)
    # c) i g)
    c = np.dot(np.linalg.inv(v4), pop)
    print("Vandermond " +str(c))
    c_round = np.dot(np.linalg.inv(v4),pop_round)
    print("Rounded "+str(c_round))
    x_vals = np.arange(1900, 1981, 1)
    y_vals = np.zeros(81)
    y_vals_round = np.zeros(81)
    for i in range(len(x_vals)):
        y_vals[i] = horner(f4_val(x_vals[i]), c)
        y_vals_round[i] = horner(f4_val(x_vals[i]), c_round)
    plt.plot(x_vals, y_vals_round)
    plt.show()
    plt.plot(x_vals, y_vals)
    plt.show()
    for i in range(len(x_vals)):
        y_vals[i] = lagrange_polynomial(years,pop , x_vals[i])
    plt.plot(x_vals, y_vals)
    plt.show()
    c_newton = coefficients(years,pop)
    print("Newton: "+str(c_newton))
    for i in range(len(x_vals)):
        y_vals[i] = newton_polynomial(c_newton,years,x_vals[i])
    plt.plot(x_vals, y_vals)
    plt.show()
