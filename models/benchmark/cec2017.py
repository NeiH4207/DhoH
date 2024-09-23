import numpy as np
from cec2017.functions import all_functions


class Functions:
    def __init__(self, name: str, dim: int):
        assert dim > 0
        self.D = dim
        self.function = all_functions[int(name[1:])-1]
        self.name = name
        self.lb = -100
        self.ub = 100

    def value(self, x: np.ndarray):
        x = x.reshape(1, -1)
        return self.function(x)
