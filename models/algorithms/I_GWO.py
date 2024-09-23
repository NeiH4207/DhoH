import random
import time
import numpy as np
from mealpy.optimizer import Optimizer

import numpy as np
from typing import List, Union, Tuple, Dict
from mealpy.utils.agent import Agent
from mealpy.utils.problem import Problem
from mealpy.utils.termination import Termination

# source: https://github.com/matikuto/Python-implementation-of-Gray-Wolf-Optimizer-and-Improved-Gray-Wolf-Optimizer

class I_GWO(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 n_limits: int = 25, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False
        
        self.n_epochs = epoch
        self.pop_size = pop_size
        
    # generate wolf
    def wolf(self, vector_size, min_range, max_range):
        wolf_position = [0.0 for i in range(vector_size)]
        for i in range(vector_size):
            wolf_position[i] = ((max_range - min_range) * random.random() + min_range)
        return wolf_position
  
    # generate wolf pack
    def pack(self):
        pack = [self.wolf(self.problem.n_dims, -10, 10) for i in range(self.pop_size)]
        return pack
    
    def evolve(self, current_epoch, current_runtime):
        alpha, beta, delta = [s.solution for s in sorted([idv for idv in self.pop],
                                                         key=lambda x: x.target.fitness)[:3]]
        
        if self.termination.max_time is not None:
            a = 2 * current_runtime / self.termination.max_time
        elif self.mode == 'epoch_bound':
            a = 2 * current_epoch / self.n_epochs
        else:
            a = 2 * self.nfe_counter / self.termination.max_fe
        
        # updating each population member with the help of alpha, beta and delta
        for i in range(self.pop_size):
            # compute A and C 
            A1, A2, A3 = a * (2 * random.random() - 1), a * (2 * random.random() - 1), a * (2 * random.random() - 1)
            C1, C2, C3 = 2 * random.random(), 2*random.random(), 2*random.random()
            
            # generate vectors for new position
            X1 = self.generate_empty_agent()
            X2 = self.generate_empty_agent()
            X3 = self.generate_empty_agent()
            X_GWO = self.generate_empty_agent()
            
            # hunting 
            for j in range(self.problem.n_dims):
                X1.solution[j] = alpha[j] - A1 * abs(C1 - alpha[j] - self.pop[i].solution[j])
                X2.solution[j] = beta[j] - A2 * abs(C2 - beta[j] - self.pop[i].solution[j])
                X3.solution[j] = delta[j] - A3 * abs(C3 - delta[j] - self.pop[i].solution[j])
                X_GWO.solution[j] += X1.solution[j] + X2.solution[j] + X3.solution[j]
            
            for j in range(self.problem.n_dims):
                X_GWO.solution[j] /= 3.0

            # fitness calculation of new position candidate
            X_GWO = self.generate_agent(X_GWO.solution)
            
            # current wolf fitness
            current_wolf = self.pop[i]
            
            # Begin i-GWO ehancement, Compute R --------------------------------
            R = current_wolf.target.fitness - X_GWO.target.fitness
            
            # Compute eq. 11, build the neighborhood
            neighborhood = []
            for l in self.pop:
                neighbor_distance = current_wolf.target.fitness - l.target.fitness
                if neighbor_distance <= R:
                    neighborhood.append(l)
                    
            # if the neigborhood is empy, compute the distance with respect 
            # to the other wolfs in the population and choose the one closer
            closer_neighbors = []
            if len(neighborhood) == 0:
                for n in self.pop:
                    distance_wolf_alone = current_wolf.target.fitness - n.target.fitness
                    closer_neighbors.append((distance_wolf_alone,n))
                    
                closer_neighbors = sorted(closer_neighbors, key=lambda x: x[0])
                neighborhood.append(closer_neighbors[0][1])
            
            # Compute eq. 12 compute new candidate using neighborhood
            X_DLH = self.generate_empty_agent()
            for m in range(self.problem.n_dims):
                random_neighbor = random.choice(neighborhood)
                random_wolf_pop = random.choice(self.pop)
                
                X_DLH.solution[m] = current_wolf.solution[m] + random.random() * random_neighbor.solution[m] - random_wolf_pop.solution[m]
            
            X_DLH = self.generate_agent(X_DLH.solution)
            
            # if X_GWO is better than X_DLH, select X_DLH
            if X_GWO.target.fitness < X_DLH.target.fitness:
                candidate = X_GWO
            else:
                candidate = X_DLH
            
            # if new position is better then replace, greedy update
            if candidate.target.fitness < self.pop[i].target.fitness:
                self.pop[i] = candidate
        
        
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