import numpy as np
from pathlib import Path
from models.benchmark.basic import *


class Functions:
    def __init__(self, name: str, dim: int, rotation=None, shift=None, shuffle=None):
        assert dim > 0
        self.name = name
        self.D = dim

        if name in ['uni1', 'uni2', 'uni3']:
            return

        self.M = rotation
        self.o = shift
        self.ss = shuffle

        indices = {
            'uni4': 1,
            'uni5': 3,
            'multi1': 4,
            'multi2': 5,
            'multi3': 6,
            'multi4': 9,
            'multi5': 10,
            'hybrid1': 11,
            'hybrid2': 12,
            'hybrid3': 14,
            'hybrid4': 15,
            'hybrid5': 18,
            'composition1': 21,
            'composition2': 22,
            'composition3': 23,
            'composition4': 24,
            'composition5': 26
        }

        base_path = Path(__file__).parent
        index = indices[name]

        if self.M is None:
            try:
                self.M = np.loadtxt(str((base_path / ('./data/2017/M_%d_D%d.txt'
                                                      % (index, dim))).resolve()))
                if 'composition' in name:
                    self.M = np.split(self.M, 8 if dim == 2 else 10)
            except:
                raise Exception('Missing rotation matrix!')
        if self.o is None:
            try:
                self.o = np.loadtxt(str((base_path / ('./data/2017/shift_data_%d.txt'
                                                      % index)).resolve()))
                if 'composition' not in name:
                    self.o = self.o[:dim]
                else:
                    self.o = self.o[:, :dim]

            except:
                raise Exception('Missing shift vector!')
        if self.ss is None and 'hybrid' in name:
            try:
                self.ss = np.loadtxt(str((base_path / ('./data/2017/shuffle_data_%d_D%d.txt'
                                                       % (index, dim))).resolve())).astype(int) - 1
            except:
                raise Exception('Missing shuffle vector!')

    def value(self, x):
        fn = getattr(self, '_Functions__' + self.name)
        return fn(x)

    # === UNI MODAL ===
    # Sphere function
    def __uni1(self, x):
        return sphere(x)

    # Schwefel 1.2 function
    def __uni2(self, x):
        return schwefel_1_2(x)

    # Rosenbrock function
    def __uni3(self, x):
        return dyxon_price(x)

    # CEC17 #1 - Shifted and Rotated Bent Cigar
    def __uni4(self, x):
        return bent_cigar(self.M @ (x - self.o)) + 100

    # CEC17 #2 - Shifted and Rotated Zakharov Function
    def __uni5(self, x):
        return zakharov(self.M @ (x - self.o)) + 300

    # === MULTI MODAL ===
    # CEC17 #3 - Shifted and Rotated Rosenbrock’s Function
    def __multi1(self, x):
        return rosenbrock(self.M @ (x - self.o)) + 400

    # CEC17 #4 - Shifted and Rotated Rastrigin’s Function
    def __multi2(self, x):
        return rastrigin(self.M @ (x - self.o)) + 500

    # CEC17 #5 - Shifted and Rotated Schaffer’s F7 Function
    def __multi3(self, x):
        return schaffer_f7(self.M @ (x - self.o)) + 600

    # CEC17 #8 - Shifted and Rotated Levy Function
    def __multi4(self, x):
        return levy(self.M @ (x - self.o)) + 900

    # CEC17 #9 - Shifted and Rotated Schwefel’s Function
    def __multi5(self, x):
        return modified_shwefel(self.M @ (x - self.o)) + 1000

    # === HYBRID ===
    # CEC17 #10 - Hybrid Function 1
    def __hybrid1(self, x):
        D = x.shape[0]
        ns = np.ceil(np.asarray([0.2, 0.6]) * D).astype(int)
        y = (self.M @ (x - self.o))[self.ss]
        z1, z2, z3 = np.split(y, ns)
        return zakharov(z1) + rosenbrock(z2) + rastrigin(z3) + 1100

    # CEC17 #11 - Hybrid Function 2
    def __hybrid2(self, x):
        D = x.shape[0]
        ns = np.ceil(np.asarray([0.3, 0.6]) * D).astype(int)
        y = (self.M @ (x - self.o))[self.ss]
        z1, z2, z3 = np.split(y, ns)
        return high_conditioned_elliptic(z1) + modified_shwefel(z2) + bent_cigar(z3) + 1200

    # CEC17 #13 - Hybrid Function 4
    def __hybrid3(self, x):
        D = x.shape[0]
        ns = np.ceil(np.asarray([0.2, 0.4, 0.6]) * D).astype(int)
        y = (self.M @ (x - self.o))[self.ss]
        z1, z2, z3, z4 = np.split(y, ns)
        return high_conditioned_elliptic(z1) + ackley(z2) + schaffer_f7(z3) + rastrigin(z4) + 1400

    # CEC17 #14 - Hybrid Function 5
    def __hybrid4(self, x):
        D = x.shape[0]
        ns = np.ceil(np.asarray([0.2, 0.4, 0.7]) * D).astype(int)
        y = (self.M @ (x - self.o))[self.ss]
        z1, z2, z3, z4 = np.split(y, ns)
        return bent_cigar(z1) + hgbat(z2) + rastrigin(z3) + rosenbrock(z4) + 1500

    # CEC17 #17 - Hybrid Function 8
    def __hybrid5(self, x):
        D = x.shape[0]
        ns = np.ceil(np.asarray([0.2, 0.4, 0.6, 0.8]) * D).astype(int)
        y = (self.M @ (x - self.o))[self.ss]
        z1, z2, z3, z4, z5 = np.split(y, ns)
        return high_conditioned_elliptic(z1) + ackley(z2) + rastrigin(z3) + hgbat(z4) + discus(z5) + 1800

    # === COMPOSITE ===
    # def __w(self, x, o, sigma):
    #     D = x.shape[0]
    #     s = np.sum(np.power(x - o, 2))
    #     if abs(s) < 1e-9:
    #         return float('inf')
    #     return 1 / math.sqrt(s) * math.exp(-s / (2 * D * sigma ** 2))

    def __composition(self, x, functions, sigmas, lambdas, biases):
        N = len(functions)
        D = x.shape[0]
        M = self.M
        o = self.o

        w = np.empty(N)
        for i in range(N):
            s = np.sum(np.power(x - o[i], 2))
            if abs(s) < 1e-9:
                w[i] = float('inf')
            w[i] = 1 / math.sqrt(s) * math.exp(-s / (2 * D * sigmas[i] ** 2))
        # w = np.asarray([self.__w(x, o[i], sigmas[i]) for i in range(N)])
        if np.max(w) == float('inf'):
            w = np.where(w == float('inf'), 1, 0)
        omega = w / np.sum(w)

        return sum([omega[i] * (lambdas[i] * functions[i](M[i] @ (x - o[i])) + biases[i])
                    for i in range(N)])

    # CEC17 #20 - Composition Function 1
    def __composition1(self, x):
        functions = [rosenbrock, high_conditioned_elliptic, rastrigin]
        sigmas = [10, 20, 30]
        lambdas = [1, 1e-6, 1]
        biases = [0, 100, 200]
        return self.__composition(x, functions, sigmas, lambdas, biases) + 2100

    # CEC17 #21 - Composition Function 2
    def __composition2(self, x):
        functions = [rastrigin, griewank, modified_shwefel]
        sigmas = [10, 20, 30]
        lambdas = [1, 10, 1]
        biases = [0, 100, 200]
        return self.__composition(x, functions, sigmas, lambdas, biases) + 2200

    # CEC17 #22 - Composition Function 3
    def __composition3(self, x):
        functions = [rosenbrock, ackley, modified_shwefel, rastrigin]
        sigmas = [10, 20, 30, 40]
        lambdas = [1, 10, 1, 1]
        biases = [0, 100, 200, 300]
        return self.__composition(x, functions, sigmas, lambdas, biases) + 2300

    # CEC17 #23 - Composition Function 4
    def __composition4(self, x):
        functions = [ackley, high_conditioned_elliptic, griewank, rastrigin]
        sigmas = [10, 20, 30, 40]
        lambdas = [10, 1e-6, 10, 1]
        biases = [0, 100, 200, 300]
        return self.__composition(x, functions, sigmas, lambdas, biases) + 2400

    # CEC17 #25 - Composition Function 6
    def __composition5(self, x):
        functions = [schaffer_f6, modified_shwefel, griewank, rosenbrock, rastrigin]
        sigmas = [10, 20, 20, 30, 40]
        lambdas = [2e-3, 1, 10, 1, 10]
        biases = [0, 100, 200, 300, 400]
        return self.__composition(x, functions, sigmas, lambdas, biases) + 2600
