import numpy as np


def lagrange_polynomial_standard(x_values, y_values, x):
    sol = 0
    t = 0
    for i in range(0, len(x_values)):
        t = 1
        for j in range(0, len(y_values)):
            if(j != i):
                t = t*((x-x_values[j])/(x_values[i]-x_values[j]))
        sol += t*y_values[i]
    return sol

#transforms chebyshew points to interval [a,b]

def transform_cheb(nodes, a, b):
    for i in range(len(nodes)):
        nodes[i] = a+(b-a)*(nodes[i]+1)/2

#calculates n standard chebyshew points on interval[1,1]
  
def get_nodes_cheb(n):
    nodes = np.empty(shape=n, dtype=float)
    for i in range(n):
        nodes[i] = np.cos(np.pi*(2*i+1)/(2*n+2))
    return nodes[i]

#gets n equidistant nodes in interval [a,b] 

def get_eqdist(n, a, b):
    nodes = np.empty(shape=n, dtype=float)
    h = (b-a)/n
    for i in range(n):
        nodes[i] = a+h*i        
    nodes[n-1] = b
    return nodes

#calculates sigmas in matrix equation (strona 17)

def get_qubic_sigmar(nodes, y_values):
    n = len(nodes) - 1
    hs = np.zeros(shape=n)
    for i in range(n):
        hs[i] = nodes[i+1] - nodes[i]
    h_matrix = np.zeros(shape=(n+1, n+1))
    h_matrix[0][0] = -hs[0]
    h_matrix[0][1] = hs[0]
    h_matrix[n-1][n-2] = hs[n-1]
    h_matrix[n][n] = -hs[n-1]
    for i in range(1,n):
        h_matrix[i][i-1] = hs[i-1]
        h_matrix[i][i] = 2*(hs[i-1]+hs[i])
        h_matrix[i][i+1] = hs[i]
    a = coefficient(nodes, y_values)
    b = coefficient(nodes[n-3:], y_values[n-3:])
    d_matrix = np.zeros(shape=n+1)
    d_matrix[0] = hs[0]*hs[0]*a[3]
    d_matrix[n] = -hs[n-1]*hs[n-1]*b[3]
    for i in range(1,n):
        d_matrix[i] = (y_values[i+1]-y_values[i])/hs[i] - (y_values[i]-y_values[i-1])/hs[i-1]
    return np.linalg.solve(h_matrix,d_matrix)

#calculates those fucked up deltas raised to power

def coefficient(x, y):
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

if __name__ == "__main__":
    x = [1,2,3,4,5,6,7,8,9,10]
    y =[30,50,70,75,78,80,90,100,105,110]
    print(get_qubic_sigmar(x,y))
    pass