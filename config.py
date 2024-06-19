import os


basedir = os.path.abspath(os.path.dirname(__file__))


class Params:
    C_PSO = { 'w': 0.9 }
    C_WOA = { 'p': 0.5, 'b': 0.1 }
    DMeSR_PSO = { 'w_max': 1.05, 'w_min': 0.5, 'c1': 1.49445, 'c2': 1.49445 }
    A_EO = {}
    GSKA = { 'p': 0.1, 'kf': 0.5, 'kr': 0.9, 'k': 10 }
    HGS = { 'L': 0.08, 'LH': 10000 }
    HI_WOA = {}
    LSHADE = { 'miu_f': 0.5, 'miu_cr': 0.5 }
    RCGA_rdn = { 's': 5, 'p_c': 1.0, 'p_m': 1.0, 'k': 5, 'K': 5 }
    SMA = { 'z': 0.03 }


class Benchmark:
    TRIALS = 10
    POP_SIZE = 100
    EPOCH = 1000
    TIME_BOUND = 20
    DIMENSIONS = 30
    OUTPUT = os.path.join(basedir, 'output')
    JOBS = -1
    LOWERBOUND = -100
    UPPERBOUND = 100

class App:
    TRIALS = 10
    POP_SIZE = 100
    EPOCH = 1000
    TIME_BOUND = 20
    DIMENSIONS = 30
    OUTPUT = os.path.join(basedir, 'output')
    JOBS = -1
    LOWERBOUND = -100
    UPPERBOUND = 100
