from numpy import ceil, sqrt, array, mean, cos, std, zeros,  subtract, sum, exp
import numpy as np
from random import choice, sample
import time
from sys import exit
from mealpy.optimizer import Optimizer

import numpy as np
from typing import List, Union, Tuple, Dict
from mealpy.utils.agent import Agent
from mealpy.utils.problem import Problem
from mealpy.utils.termination import Termination

import scipy.stats
from models.algorithms.utils import crossover, current_to_pbest_weighted_mutation, selection
EPS = 1e-6

class JSO(Optimizer):

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 n_limits: int = 25, **kwargs: object) -> None:
        super().__init__( **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False
        self.pop_size = pop_size
        
    def run(self, pop, m_cr, m_f, k,  archive, memory_size, memory_indexes, p, p_max, p_min,
        current_epoch, current_runtime):
        fitness = np.array([item.target.fitness for item in pop]).reshape(-1)
        # 2.1 Adaptation
        r = np.random.choice(memory_indexes, len(pop))
        m_cr[- 1] = 0.9
        m_f[-1] = 0.9
        cr = np.random.normal(m_cr[r], 0.1, len(pop))
        cr = np.clip(cr, 0, 1)
        cr[m_cr[r] == 1] = 0
        cr[m_cr[r] < 0] = 0
        if self.get_progress_percentage(current_epoch, current_runtime) < 0.25:
            cr[cr < 0.7] = 0.7
        elif self.get_progress_percentage(current_epoch, current_runtime) < 0.5:
            cr[cr < 0.6] = 0.6
        f = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=len(pop))
        while sum(f <= 0) != 0:
            r = np.random.choice(memory_indexes, sum(f <= 0))
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=sum(f <= 0))

        f = np.clip(f, 0, 1)
        if self.get_progress_percentage(current_epoch, current_runtime) < 0.6:
            f = np.clip(f, 0, 0.7)
        
        
        # 2.2 Common steps
        # 2.2.1 Calculate weights for mutation
        weighted = f.copy().reshape(len(f), 1)

        if self.get_progress_percentage(current_epoch, current_runtime) < 0.2:
            weighted *= .7
        elif self.get_progress_percentage(current_epoch, current_runtime) < 0.4:
            weighted *= .8
        else:
            weighted *= 1.2

        weighted = np.clip(weighted, 0, 1)
        # print(min(fitness), min(cr), max(cr), min(f), max(f))
        mutated = current_to_pbest_weighted_mutation(pop, fitness, f.reshape(len(f), 1),
                                                                weighted, p, np.array([self.lb, self.ub]).T)
        crossed = crossover(pop, mutated, cr.reshape(len(f), 1))
        c_fitness = np.array([item.target.fitness for item in crossed]).reshape(-1)

        pop, indexes = selection(pop, crossed, fitness, c_fitness, return_indexes=True)

        # 2.3 Adapt for next generation
        archive.extend(pop[indexes])

        if len(indexes) > 0:
            if len(archive) > self.pop_size:
                archive = sample(archive, self.pop_size)

            weights = np.abs(fitness[indexes] - c_fitness[indexes])
            weights /= np.sum(weights)

            if max(cr) != 0:
                m_cr[k] = (np.sum(weights * cr[indexes]**2) / np.sum(weights * cr[indexes]) + m_cr[-1]) / 2
            else:
                m_cr[k] = 1

            m_f[k] = np.sum(weights * f[indexes]**2) / np.sum(weights * f[indexes])

            k += 1
            if k == memory_size:
                k = 0

        fitness[indexes] = c_fitness[indexes]
        # Adapt population size
        new_population_size = round((4 - self.pop_size) / self.fes_bound * self.fes + self.pop_size)
        if self.pop_size > new_population_size:
            self.pop_size = new_population_size
            best_indexes = np.argsort(fitness)[:self.pop_size]
            pop = pop[best_indexes]
            fitness = fitness[best_indexes]
            if k == memory_size:
                k = 0

        # Adapt p
        p = (p_max - p_min) / self.fes_bound * self.fes + p_min
        return pop, [pop[np.argmin(fitness)], np.min(fitness)]
    
    def get_progress_percentage(self, current_epoch, current_runtime):
        if self.termination.max_time is not None:
            return current_runtime / self.termination.max_time
        elif self.mode == 'epoch_bound':
            return current_epoch / self.termination.max_epoch
        else:
            return self.nfe_counter / self.termination.max_fe
        
    def evolve(self, current_epoch, current_runtime):
        pop = self.pop
        fitness = np.array([item.target.fitness for item in pop]).reshape(-1)
        # 2.1 Adaptation
        r = np.random.choice(self.memory_indexes, len(pop))
        k = self.k
        self.m_cr[- 1] = 0.9
        self.m_f[-1] = 0.9
        cr = np.random.normal(self.m_cr[r], 0.1, len(pop))
        cr = np.clip(cr, 0, 1)
        cr[self.m_cr[r] == 1] = 0
        cr[self.m_cr[r] < 0] = 0
        if self.get_progress_percentage(current_epoch, current_runtime) < 0.25:
            cr[cr < 0.7] = 0.7
        elif self.get_progress_percentage(current_epoch, current_runtime) < 0.5:
            cr[cr < 0.6] = 0.6
        f = scipy.stats.cauchy.rvs(loc=self.m_f[r], scale=0.1, size=len(pop))
        while sum(f <= 0) != 0:
            r = np.random.choice(self.memory_indexes, sum(f <= 0))
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=self.m_f[r], scale=0.1, size=sum(f <= 0))

        f = np.clip(f, 0, 1)
        if self.get_progress_percentage(current_epoch, current_runtime) < 0.6:
            f = np.clip(f, 0, 0.7)
        
        
        # 2.2 Common steps
        # 2.2.1 Calculate weights for mutation
        weighted = f.copy().reshape(len(f), 1)

        if self.get_progress_percentage(current_epoch, current_runtime) < 0.2:
            weighted *= .7
        elif self.get_progress_percentage(current_epoch, current_runtime) < 0.4:
            weighted *= .8
        else:
            weighted *= 1.2

        weighted = np.clip(weighted, 0, 1)
        # print(min(fitness), min(cr), max(cr), min(f), max(f))
        pop_position = np.array([item.solution for item in pop])
        mutated = current_to_pbest_weighted_mutation(pop_position, fitness, f.reshape(len(f), 1),
                                                                weighted, self.p, np.array([self.problem.lb, self.problem.ub]).T)
        crossed = crossover(pop_position, mutated, cr.reshape(len(f), 1))
        crossed_pop = [self.generate_agent(crossed[i]) for i in range(len(crossed))]
        c_fitness = np.array([item.target.fitness for item in crossed_pop]).reshape(-1)

        pop_position, indexes = selection(pop_position, crossed, fitness, c_fitness, return_indexes=True)
        
        pop = [self.generate_agent(pop_position[i]) for i in range(len(pop_position))]

        # 2.3 Adapt for next generation
        self.archive.extend([pop[i] for i in indexes])

        if len(indexes) > 0:
            if len(self.archive) > self.pop_size:
                self.archive = sample(self.archive, self.pop_size)

            weights = np.abs(fitness[indexes] - c_fitness[indexes])
            weights /= np.sum(weights)

            if max(cr) != 0:
                self.m_cr[k] = (np.sum(weights * cr[indexes]**2) / np.sum(weights * cr[indexes]) + self.m_cr[-1]) / 2
            else:
                self.m_cr[k] = 1

            self.m_f[k] = np.sum(weights * f[indexes]**2) / np.sum(weights * f[indexes])

            k += 1
            if k == self.memory_size:
                k = 0

        fitness[indexes] = c_fitness[indexes]
        # Adapt population size
        
        
        if self.termination.max_time is not None:
            new_population_size = round((4 - self.pop_size) / self.termination.max_time * current_runtime + self.pop_size)
        elif self.mode == 'epoch_bound':
            new_population_size = round((4 - self.pop_size) / self.termination.max_epoch * current_epoch + self.pop_size)
        else:
            new_population_size = round((4 - self.pop_size) / self.termination.max_fe * self.nfe_counter + self.pop_size)
        
        if self.pop_size > new_population_size:
            self.pop_size = new_population_size
            best_indexes = np.argsort(fitness)[:self.pop_size]
            pop = [pop[i] for i in best_indexes]
            fitness = fitness[best_indexes]
            if k == self.memory_size:
                k = 0

        # Adapt p
        self.k = k
        if self.termination.max_time is not None:
            self.p = (self.p_max - self.p_min) / self.termination.max_time * current_runtime + self.p_min
        elif self.mode == 'epoch_bound':
            self.p = (self.p_max - self.p_min) / self.termination.max_epoch * current_epoch + self.p_min          
        else:
           self.p = (self.p_max - self.p_min) / self.termination.max_fe * self.nfe_counter + self.p_min
        
        self.pop = pop
        
        
    def solve(self, problem: Union[Dict, Problem] = None, mode: str = 'single', n_workers: int = None,
              termination: Union[Dict, Termination] = None, starting_solutions: Union[List, np.ndarray, Tuple] = None,
              seed: int = None) -> Agent:
        """
        Args:
            problem: an instance of Problem class or a dictionary
            mode: Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                * 'process': The parallel mode with multiple cores run the tasks
                * 'thread': The parallel mode with multiple threads run the tasks
                * 'swarm': The sequential mode that no effect on updating phase of other agents
                * 'single': The sequential mode that effect on updating phase of other agents, this is default mode

            n_workers: The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
            termination: The termination dictionary or an instance of Termination class
            starting_solutions: List or 2D matrix (numpy array) of starting positions with length equal pop_size parameter
            seed: seed for random number generation needed to be *explicitly* set to int value

        Returns:
            g_best: g_best, the best found agent, that hold the best solution and the best target. Access by: .g_best.solution, .g_best.target
        """
        self.check_problem(problem, seed)
        self.check_mode_and_workers(mode, n_workers)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_solutions)
        self.initialization()
        self.after_initialization()

        self.before_main_loop()
        start_time = time.perf_counter()
        
        self.m_cr = np.ones(self.pop_size) * .8
        self.m_f = np.ones(self.pop_size) * .5
        
        self.archive = []
        self.k = 0
        self.memory_size = self.pop_size
        self.memory_indexes = list(range(self.memory_size))
        self.p_max = .25
        self.p_min = self.p_max / 2
        self.p = self.p_max
        
        for epoch in range(1, self.epoch + 1):
            time_epoch = time.perf_counter()

            ## Evolve method will be called in child class
            self.evolve(epoch, time_epoch - start_time)

            # Update global best solution, the population is sorted or not depended on algorithm's strategy
            pop_temp, self.g_best = self.update_global_best_agent(self.pop)
            if self.sort_flag: self.pop = pop_temp

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch, time_epoch)
            if self.check_termination("end", None, epoch):
                break
        self.track_optimize_process()
        return self.g_best
        