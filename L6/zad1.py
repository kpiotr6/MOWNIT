import numpy as np
import scipy
from matplotlib import pyplot as plt
from threading import Thread


def f(x):
    return 4 / (1 + x ** 2)


def rect_integrate(f: callable, a, b, num_of_nodes):
    n = num_of_nodes
    args = np.linspace(a, b, n, dtype=np.float64)

    summ = 0
    for i in range(1, n):
        x2 = args[i]
        x1 = args[i - 1]
        summ += (x2 - x1) * f((x1 + x2) / 2)

    return summ


EPSILON = 2 ** -52

error_rect = np.zeros(25, dtype=np.float64)
error_simps = np.zeros(25, dtype=np.float64)
error_trapez = np.zeros(25, dtype=np.float64)

steps = np.zeros(25, dtype=np.float64)

a = 0
b = 1


# for m in range(1, 26):
#     print(m)
#     num_of_nodes = 2 ** m + 1
#     args = np.linspace(a, b, num_of_nodes, dtype=np.float64)
#     vals = np.array([f(x) for x in args], dtype=np.float64)

#     err1 = abs(rect_integrate(f, 0, 1, num_of_nodes) - np.pi) / np.pi
#     err2 = abs(scipy.integrate.trapz(vals, args) - np.pi) / np.pi
#     err3 = abs(scipy.integrate.simps(vals, args) - np.pi) / np.pi

#     error_rect[m - 1] = err1
#     error_trapez[m - 1] = err2
#     error_simps[m - 1] = err3

#     steps[m - 1] = 1 / num_of_nodes


error_rect = np.array([6.60820478e-03, 1.65771485e-03, 4.14463664e-04, 1.03616463e-04,
                       2.59041243e-05, 6.47603120e-06, 1.61900780e-06, 4.04751950e-07,
                       1.01187987e-07, 2.52969968e-08, 6.32424885e-09, 1.58106165e-09,
                       3.95265765e-10, 9.88133315e-11, 2.47047111e-11, 6.16999337e-12,
                       1.54066069e-12, 4.07535073e-13, 1.00788244e-13, 1.27222187e-15,
                       4.09938159e-15, 1.09269723e-13, 1.36269098e-13, 5.42814666e-14,
                       2.09068461e-13], dtype=np.float64)

error_trapez = np.array([1.32393528e-02, 3.31557403e-03, 8.28929586e-04, 2.07232961e-04,
                         5.18082491e-05, 1.29520624e-05, 3.23801561e-06, 8.09503902e-07,
                         2.02375975e-07, 5.05939938e-08, 1.26484984e-08, 3.16212443e-09,
                         7.90531248e-10, 1.97632741e-10, 4.94082914e-11, 1.23518608e-11,
                         3.08796520e-12, 7.71955961e-13, 1.93660441e-13, 4.83444312e-14,
                         1.27222187e-14, 3.10987569e-15, 2.54444375e-15, 2.68580173e-15,
                         2.68580173e-15], dtype=np.float64)

error_simps = np.array([2.62902329e-03, 7.64775751e-06, 4.81065190e-08, 7.52793755e-10,
                        1.17638116e-11, 1.83765382e-13, 2.82715972e-15, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.41357986e-16,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.41357986e-16,
                        2.82715972e-16, 1.41357986e-16, 4.24073958e-16, 1.55493784e-15,
                        2.12036979e-15], dtype=np.float64)

print(error_rect)
print(error_trapez)
print(error_simps)

h_min = np.sqrt(EPSILON)
min_step = 1 / (2 ** 20 + 1)  # m = 20
print("h_min:", h_min)
print("Min step:", min_step)  # m = 20

m_values = np.array([m for m in range(1, 26)])

plt.plot(m_values, error_rect, c="red", label="Error rectangle")
plt.plot(m_values, error_trapez, c="green", label="Error trapez")
plt.plot(m_values, error_simps, c="yellow", label="Error Simpson")

plt.yscale("log")
plt.legend()

plt.show()


def convergence(m1, m2, errors):
    return np.log(errors[m2] / errors[m1]) / np.log((1 / (2 ** m2 + 1)) / (1 / (2 ** m1 + 1)))


print("Rząd zbieżności dla metody prostokątów:", convergence(4, 5, error_rect))
print("Rząd zbieżności dla metody trapezów:", convergence(4, 5, error_trapez))
print("Rząd zbieżności dla metody Simpsona:", convergence(4, 5, error_simps))


def legend_integrate(f, nodes, weights):
    summ = np.dot(f(nodes), weights)
    return summ


t = 16
error_leggaus = np.zeros(t, dtype=np.float64)

# for i in range(1, t):
#     n = 2 ** i + 1
#     xd = scipy.special.roots_legendre(n)
#     weights_scaled = np.array([((1 - 0) / (1 - (-1))) * w for w in xd[1]])
#     node_scaled = np.array([((1 - 0) * x + 0 * (-1) - 1 * (-1)) / (1 - (-1)) for x in xd[0]], dtype=np.float64)
#     val = legend_integrate(f, node_scaled, weights_scaled)
#     print(i, ",", val)
#     error_leggaus[i - 1] = abs(np.pi - val) / np.pi


print(error_leggaus)

error_leggaus = np.array([1.66957873e-04, 4.36244979e-09, 4.93480729e-13, 1.41357986e-16,
                          1.41357986e-16, 0.00000000e+00, 1.41357986e-16, 5.65431943e-16,
                          8.48147915e-16, 0.00000000e+00, 3.67530763e-15, 1.48425885e-14,
                          2.04969079e-14, 1.11672809e-14, 2.95438190e-14, 0.00000000e+00], dtype=np.float64)

i_values = np.array([i for i in range(t)])

plt.plot(m_values, error_rect, c="red", label="Error rectangle")
plt.plot(m_values, error_trapez, c="green", label="Error trapez")
plt.plot(m_values, error_simps, c="yellow", label="Error Simpson")
plt.plot(i_values, error_leggaus, c="pink", label="Error Gauss-Legendre")

plt.yscale("log")
plt.legend()

plt.show()
