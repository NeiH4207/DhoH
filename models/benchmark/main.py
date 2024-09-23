import numpy as np

from functions import Functions
from models.algorithms import EO, GSKA, HGS, SMA, LSHADE


if __name__ == '__main__':
    x = np.random.uniform(-100, 100, 50)
    f1 = Functions('uni1', 100)
    print(f1.value(x))

    lowerbound = [-100, ] * 10
    upperbound = [100, ]*10
    model1 = EO.BaseEO(f1.value, lb=lowerbound, ub=upperbound, verbose=True, epoch=750, pop_size=100)
    best_position, best_fitness, loss_fitness_training = model1.train()

    model2 = GSKA.OriginalGSKA(f1.value, lb=lowerbound, ub=upperbound, verbose=True, epoch=750, pop_size=100)
    best_position, best_fitness, loss_fitness_training = model2.train()

    model3 = HGS.OriginalHGS(f1.value, lb=lowerbound, ub=upperbound, verbose=True, epoch=750, pop_size=100)
    best_position, best_fitness, loss_fitness_training = model3.train()

    model4 = SMA.BaseSMA(f1.value, lb=lowerbound, ub=upperbound, verbose=True, epoch=750, pop_size=100)
    best_position, best_fitness, loss_fitness_training = model4.train()

    model5 = LSHADE.L_SHADE(f1.value, lb=lowerbound, ub=upperbound, verbose=True, epoch=750, pop_size=100)
    best_position, best_fitness, loss_fitness_training = model5.train()


