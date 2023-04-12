# import numpy as np
# import matplotlib.pyplot as plt

# def f(x):
#     return np.sqrt(x)

# n = 1000 # degree of polynomial
# x = np.cos((np.arange(n)+0.5)*np.pi/n)*1.0 + 1.0
# V = np.zeros((n,n))
# for i in range(n):
#     V[:,i] = np.cos(i*np.arccos(x))
# W = np.diag(np.ones(n))
# W[0,0] = 0.5
# W[n-1,n-1] = 0.5
# y = f(x)
# c = np.linalg.solve(V.T @ W @ V, V.T @ W @ y)

# def g(x):
#     s = 0
#     for i in range(n):
#         s += c[i]*np.cos(i*np.arccos(x))
#     return s

# xx = np.linspace(0, 2, 1000)
# plt.plot(xx, f(xx), 'b-', label='f(x)')
# plt.plot(xx, g(xx), 'r--', label='g(x)')
# plt.legend()
# plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt

# Define the function to be approximated
def f(x):
    return np.sqrt(x)

# Define the interval and the number of Chebyshev nodes
a, b = 0, 2
n = 3

# Define the Chebyshev nodes and the Chebyshev polynomials
x = np.zeros(n)
T = np.zeros((n, n))

for i in range(n):
    x[i] = 0.5 * (a + b) + 0.5 * (b - a) * math.cos(math.pi * (i + 0.5) / n)
    for j in range(n):
        T[i, j] = math.cos(j * math.acos((2 * x[i] - a - b) / (b - a)))

# Define the matrix A and the vector b for the linear system Ax = b
A = np.dot(T.T, T)
b = np.dot(T.T, f(x))

# Solve the linear system to obtain the coefficients of the Chebyshev approximation
c = np.linalg.solve(A, b)

# Define the Chebyshev approximation
def p(x):
    return c[0] / 2 + c[1] * x + c[2] * (2 * x**2 - 1) / 2

# Plot the original function and its Chebyshev approximation
X = np.linspace(a, b, 1000)
Y = f(X)
P = p(X)

plt.plot(X, Y, label='f(x)')
plt.plot(X, P, label='p(x)')
plt.legend()
plt.show()

