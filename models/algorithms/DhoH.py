from copy import deepcopy
from numpy.core.fromnumeric import cumsum
from numpy.random import random, normal, uniform
from numpy import  exp, sum, cos, arctanh, pi
import numpy as np
from random import randint
from bisect import bisect_left
from math import gamma, sin, sqrt
from mealpy.optimizer import Optimizer

import numpy as np
from typing import List, Union, Tuple, Dict
from mealpy.utils.agent import Agent
from mealpy.utils.problem import Problem
from mealpy.utils.termination import Termination
import os
import time

class DhoH(Optimizer):
    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 n_limits: int = 25, alpha=0.5, drop_rate=0.2,
                 r1:int =1.5, r2: int=1.25, **kwargs: object) -> None:
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
        
        self.selected_counter = np.ones(self.pop_size)
        self.n_candidates = int(0.05 * self.pop_size)
        self.drop_rate = drop_rate
        self.alpha = alpha
        self.r1 = r1
        self.r2 = r2
        
        self.history = {
            'a': [],
            'b': [],
            'c': [],
            'l': [],
        }
        
        self.distance = np.zeros((self.pop_size, self.pop_size))
        
    def sort2(self, arr):
        if (arr[1].target.fitness < arr[0].target.fitness):
            arr[0], arr[1] = arr[1], arr[0]
        return arr
    
    def sort3(self, arr):
        
        # Insert arr[1]
        if (arr[1].target.fitness < arr[0].target.fitness):
            arr[0], arr[1] = arr[1], arr[0]
            
        # Insert arr[2]
        if (arr[2].target.fitness < arr[1].target.fitness):
            arr[1], arr[2] = arr[2], arr[1]
            if (arr[1].target.fitness < arr[0].target.fitness):
                arr[1], arr[0] = arr[0], arr[1]
        return arr
    
    def sort_all(self, arr):
        if len(arr) == 2:
            arr = self.sort2(arr)
        if len(arr) == 3:
            arr = self.sort3(arr)
        else:
            arr = sorted(arr, key=lambda x: x.target.fitness)
        return arr
    
    def update_acceleration(self, acceleration, searching_fits):
        if len(searching_fits) > 2:
            kl = self.get_kl_divergence(searching_fits[-1], searching_fits[-2], searching_fits[-3])
            if kl < 1:
                if acceleration < 0.2 and kl < 0:
                    return acceleration
                return acceleration / self.r2
            else:
                return acceleration * self.r1

        return acceleration
    
    def get_random_id(self, current_size, k=1, probs=None):
        rand_ids = []
        if probs is not None:
            probs = probs / sum(probs)
            cumsum_probs = cumsum(probs)
            for i in range(k):
                rand_id = bisect_left(cumsum_probs, random())
                if rand_id not in rand_ids:
                    rand_ids.append(rand_id)
        else:
            while len(rand_ids) < k:
                rand_id = randint(0, current_size - 1)
                if rand_id not in rand_ids:
                    rand_ids.append(rand_id)
        return rand_ids
        
    def get_random_groups(self, pop, K = 2):
        counter = self.selected_counter ** 3
        selected_probs = (max(counter) - counter + 0.1) / (max(counter) - min(counter) + 0.1)
        selected_probs = selected_probs / sum(selected_probs) 
        
        rd_indv_idx = self.get_random_id(self.pop_size, 1, selected_probs)[0]
        rd_indv = pop[rd_indv_idx]
        
        pop_solution = np.array([x.solution for x in pop])
        
        diversities = np.linalg.norm(pop_solution - rd_indv.solution, axis=0)
        
        n_rd_considered_dims = np.random.randint(1, self.problem.n_dims)
        
        highest_diversity_dims = np.argsort(diversities)[-n_rd_considered_dims:]
        
        pop_considered_dim_positions = [x.solution[highest_diversity_dims] for x in pop]
        
        closest_neighbor_indices = np.argsort(np.linalg.norm(pop_considered_dim_positions - 
                                        rd_indv.solution[highest_diversity_dims], axis=1))[:K] # include itself
        closest_neighbors = [deepcopy(pop[i]) for i in closest_neighbor_indices]
        self.selected_counter[closest_neighbor_indices] += 1
        return closest_neighbors, highest_diversity_dims
    
    def get_kl_divergence(self, p, q, r):
        if (p - r) == 0:
            return 1
        return (p - q) / (p - r)
    
    def huntting_search(self, group, considered_dims):
        assert len(group) >= 2, "group must have least 2 dholes"
        acceleration = 1.0
        searching_fits = []
        early_stop = 0
        t = 0
        # sort group by fitness
        group = self.sort_all(group)
        prey = group[0]
        best = deepcopy(prey)
        dholes = group[1:]
        no_considered_dims = [i for i in range(self.problem.n_dims) if i not in considered_dims]
        
        while early_stop < 5:
            maxFit = max([x.target.fitness for x in dholes])
            minFit = prey.target.fitness
            scaled_dy = [(x.target.fitness - minFit) / (maxFit - minFit + 1e-16) for x in dholes]
            sum_fit = sum(scaled_dy) + 1e-16
            probs = [x / sum_fit for x in scaled_dy]
            searching_fits.append(prey.target.fitness)
            
            p = 1 / len(dholes)
            # add dirichlet noise with min_explore_rate
            dirictlet_rd = np.random.dirichlet(p * np.ones(len(probs)))
            exp_ratio = p / np.sqrt(t + 1)
            probs = [x * (1 - exp_ratio) + y * exp_ratio for x, y in zip(probs, dirictlet_rd)]

            new_dhole_positions = []

            for i in range(len(dholes)):
                grad_vector = probs[i] * (prey.solution - dholes[i].solution)
                # clip 0.1 * [lb, ub]
                grad_vector = np.clip(grad_vector, -0.01 * (self.problem.ub - self.problem.lb),
                                      0.01 * (self.problem.ub - self.problem.lb))
                drop_grad = np.random.choice(range(self.problem.n_dims), int(self.problem.n_dims * self.drop_rate))
                grad_vector[no_considered_dims] = 0
                grad_vector[drop_grad] = 0
                prey.solution = np.add(prey.solution, acceleration * grad_vector)
                new_pos = np.add(dholes[i].solution, grad_vector / 2)
                new_pos = self.correct_solution(new_pos)
                new_dhole_positions.append(new_pos)
            
            prey.solution = self.correct_solution(prey.solution)
            prey.target = self.get_target(prey.solution)
            searching_fits.append(prey.target.fitness)

            acceleration = self.update_acceleration(acceleration, searching_fits)

            for i in range(len(dholes)):
                dholes[i].solution = new_dhole_positions[i]
                dholes[i].target = self.get_target(dholes[i].solution)
                
            group = self.sort_all([prey] + dholes)
            prey = group[0]
            
            if best.target.fitness > prey.target.fitness:
                best = deepcopy(prey)
                early_stop = 0
            else:
                # update prey with best using alpha
                prey.solution = np.add(prey.solution, self.alpha * (best.solution - prey.solution))
                early_stop += 1
                
            t += 1
                
        return prey
    
    def levy_flight_2_random_step(self, position=None, g_best_position=None):
        alpha = 0.01
        xichma_v = 1
        xichma_u = ((gamma(1 + 1.5) * sin(pi * 1.5 / 2)) / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
        levy_b = (normal(0, xichma_u ** 2)) / (sqrt(abs(normal(0, xichma_v ** 2))) ** (1.0 / 1.5))
        return position + alpha * levy_b * normal(0., 0.2, self.problem.n_dims) * abs(position - g_best_position)
    
    def get_params(self, epoch, current_runtime):
        if self.termination.max_time is not None:
            a = arctanh(-((current_runtime + 1) / self.termination.max_time) + 1)
            b = 1 - (current_runtime + 1) / self.termination.max_time
            c = a / arctanh((self.termination.max_time - 1) / self.termination.max_time)
            l = cos(pi / 3 * (current_runtime + 1) / self.termination.max_time)
        elif self.mode == 'epoch_bound':
            a = arctanh(-((epoch + 1) / self.termination.max_epoch) + 1)
            b = 1 - (epoch + 1) / self.termination.max_epoch
            c = a / arctanh((self.termination.max_epoch - 1) / self.termination.max_epoch)
            l = cos(pi / 3 * (epoch + 1) / self.termination.max_epoch)
        else:
            a = arctanh(-((self.nfe_counter + 1) / self.termination.max_fe) + 1)
            b = 1 - (self.nfe_counter + 1) / self.termination.max_fe
            c = a / arctanh((self.termination.max_fe - 1) / self.termination.max_fe)
            l = cos(pi / 3 * (self.nfe_counter + 1) / self.termination.max_fe)
            
        return a, b, c, l
    
    def evolve(self, epoch, current_runtime):     
        a, b, c, l = self.get_params(epoch, current_runtime)
        pop = sorted(self.pop, key=lambda item: item.target.fitness)
        elites = pop[:int(self.pop_size * 0.1)]
        self.selected_counter = np.ones(self.pop_size)
        new_pop = []
        
        while len(new_pop) < self.pop_size * 0.775:
            n_neighbors = np.random.randint(2, max(2, int(np.sqrt(self.pop_size // 3))))
            rd_dhole_group, considered_dims = self.get_random_groups(pop, K = n_neighbors)
            best = self.huntting_search(rd_dhole_group, considered_dims)
            new_pop.append(best)
        
        new_pop.extend(elites)
        new_pop = sorted(new_pop, key=lambda x: x.target.fitness)
        
        list_fitness = [indv.target.fitness for indv in new_pop]
        tmp_indv = deepcopy(new_pop[0])
        current_size = len(new_pop)
        candidates = [new_pop[i] for i in range(self.n_candidates)]
        explore_rate = max(0.1, b * l)
        w = min(0.95, b * exp(a / 2) * 0.775)
        
        while len(new_pop) < self.pop_size:
            idx_selected = self.get_index_roulette_wheel_selection(list_fitness)
            rd_idv = new_pop[idx_selected]
            
            if uniform() < explore_rate: # exploration
                mutated = self.generator.uniform(self.problem.lb, self.problem.ub)
                idx_selected_2 = self.get_random_id(current_size, 1)[0]
                
                while idx_selected_2 == idx_selected:
                    idx_selected_2 = self.get_random_id(current_size, 1)[0]
                    
                rd_idv_2 = new_pop[idx_selected_2]
                child_pos = deepcopy(rd_idv.solution)
                
                for i in range(self.problem.n_dims):
                    if uniform() < 0.25:
                        child_pos[i] = rd_idv_2.solution[i]
                        
                child_pos = np.where(self.generator.random(self.problem.n_dims) < w * 0.25, mutated, child_pos)
                child_pos = self.correct_solution(child_pos)
                child = self.generate_empty_agent(child_pos)
                child.target = self.get_target(child_pos)
            else: # exploitation
                best_idx = self.get_index_roulette_wheel_selection(list_fitness[:self.n_candidates])
                candidate = candidates[best_idx]
                child_pos = candidate.solution + (1 - c) * (candidate.solution - rd_idv.solution) * normal(0, 0.01, self.problem.n_dims)
                child_pos = self.correct_solution(child_pos)
                child = deepcopy(tmp_indv)
                child = self.generate_empty_agent(child_pos)
                child.target = self.get_target(child_pos)

            current_size += 1
            new_pop.append(child)

        self.pop = new_pop
        
        
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
        