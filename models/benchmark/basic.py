import math
import numpy as np


# f1
def bent_cigar(x):
    return x[0]**2 + 1e6 * np.sum(np.square(x[1:]))


# f3
def zakharov(x):
    D = x.shape[0]
    g = 0.5 * np.sum(np.linspace(1, D, D) * x)
    return np.sum(np.square(x)) + g ** 2 + g ** 4


# f4
def rosenbrock(x):
    x = 2.048 / 100 * x + 1
    return np.sum(100 * np.square(np.square(x[:-1]) - x[1:]) + np.square(x[:-1] - 1))


# f5
def rastrigin(x):
    x = 5.12 / 100 * x
    return np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x) + 10)


# f6
def schaffer_f6(x):
    s = np.square(x) + np.square(np.roll(x, -1))   # x^2 + y^2
    g = 0.5 + (np.square(np.sin(np.sqrt(s))) - 0.5) / np.square(1 + 0.001 * s)
    return np.sum(g)


# f9
def levy(x):
    w = 1 + (x - 1) / 4
    return math.sin(math.pi * w[0]) ** 2 \
           + np.sum(np.square(w[:-1] - 1) * (1 + 10 * np.square(np.sin(np.pi * w[:-1] + 1)))) \
           + (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)


# f10
def modified_shwefel(x):
    x = 1000 / 100 * x
    D = x.shape[0]
    z = x + 4.209687462275036e2
    zz = np.abs(z)
    g = np.where(zz <= 500, z * np.sin(np.sqrt(zz)), z)
    g = np.where(z > 500, (500 - np.mod(z, 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(z, 500))))
                 - np.square(z - 500) / (10000 * D), g)
    g = np.where(z < -500, (np.mod(zz, 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(zz, 500) - 500)))
                 - np.square(z + 500) / (10000 * D), g)
    # g = np.where(zz <= 500, z * np.sin(np.sqrt(zz)),
    #              np.sign(z) * (500 - np.mod(zz, 500)) * np.sin(np.sqrt(500 - np.mod(zz, 500)))) \
    #     - np.power((zz - 500) / (10000 * D), 2)
    return 418.9829 * D - np.sum(g)


# f11
def high_conditioned_elliptic(x):
    D = x.shape[0]
    return np.sum(np.power(1e6, np.linspace(0, D-1, D) / (D-1)) * np.square(x))


# f12
def discus(x):
    return 1e6 * (x[0] ** 2) + np.sum(np.square(x[1:]))


# f13
def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.square(x)))) \
           - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.exp(1)


# f14
def weierstrass(x, a=0.5, b=3, kmax=20):
    x = 0.5 / 100 * x
    D = x.shape[0]
    k = np.linspace(0, kmax, kmax+1)
    return np.sum(np.power(a, k) * np.sum(np.cos(2 * math.pi * np.outer(np.power(b, k), x + 0.5)), axis=1)) \
           - D * np.sum(np.power(a, k) * np.cos(math.pi * np.power(b, k)))


# f15
def griewank(x):
    x = 600 / 100 * x
    D = x.shape[0]
    return np.sum(np.square(x) / 4000) - np.prod(np.cos(x / np.sqrt(np.linspace(1, D, D)))) + 1


# f18
def hgbat(x):
    x = 5 / 100 * x - 1
    D = x.shape[0]
    return math.sqrt(math.fabs(np.sum(np.square(x)) ** 2 - np.sum(x) ** 2)) \
           + (0.5 * np.sum(np.square(x) + np.sum(x))) / D + 0.5


# f20
def schaffer_f7(x):
    D = x.shape[0]
    s = np.sqrt(np.square(x[:-1]) + np.square(x[1:]))
    return (np.sum(np.sqrt(s) * (np.sin(50 * np.power(s, 0.2)) + 1)) / (D-1)) ** 2


def schwefel_1_2(x):
    return np.sum([np.sum(x[:i]) ** 2 for i in range(len(x))])


def sphere(x):
    return np.sum(np.square(x))


def dyxon_price(x):
    x = 10 / 100 * x
    D = x.shape[0]
    return (x[0] - 1) ** 2 + np.sum(np.linspace(2, D, D-1) * np.square(2 * np.square(x[1:]) - x[:-1]))
