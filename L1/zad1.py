import mpmath
import matplotlib.pyplot as plt
import numpy as np
def derf(x,h,f):
    return (f(x+h)-f(x))/h

def ders(x,h,f):
    return (f(x+h)-f(x-h))/(2*h)

def error(x,k,t,f):
    return abs(mpmath.sec(x)**2-t(x,10**-k,f))

if __name__ == "__main__":
    x = []
    y = []
    z = []
    for k in range(0,17):
        h = 10**-k
        x.append(h)
        y.append((error(1,k,derf,mpmath.tan),derf(1,10**-k,mpmath.tan),h))
        z.append((error(1,k,ders,mpmath.tan),ders(1,10**-k,mpmath.tan),h))
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x,[e[0] for e in y],color='r',label='Podstawowy wzór',marker="o")
    plt.plot(x,[e[0] for e in z],color='b',label='Różnice centralne',marker="o")
    plt.xlabel("h")
    plt.ylabel("Błąd")
    y.sort()
    z.sort()
    print("Pochodna (pierwszy wzór)")
    print(str(y[0][1])+"+-"+str(y[0][0]))
    print("Różnice centralne")
    print(str(z[0][1])+"+-"+str(z[0][0]))
    print("Wartość rzeczywista")
    print(mpmath.sec(1)**2)
    print("Pierwiastek epsilon maszynowy")
    print(mpmath.sqrt(np.finfo(float).eps))
    print("Pierwiastek sześcienny epsilon maszynowy")
    print(mpmath.nthroot(np.finfo(float).eps, 3))
    plt.legend()
    plt.show()