import os
from joblib import Parallel, delayed
from argparse import ArgumentParser

from pandas.core.frame import DataFrame

from models.algorithms.DhoH import DhoH
from mealpy.bio_based.BBO import DevBBO
from mealpy.swarm_based.GWO import RW_GWO
from mealpy.evolutionary_based.SHADE import L_SHADE
from mealpy.swarm_based.ABC import OriginalABC
from mealpy.swarm_based.WOA import HI_WOA
from mealpy.evolutionary_based.GA import EliteMultiGA
from mealpy.system_based.AEO import AugmentedAEO
from mealpy.bio_based.SMA import DevSMA
from models.algorithms.I_GWO import I_GWO
from models.algorithms.JSO import JSO

from mealpy import FloatVar
from mealpy.utils.termination import Termination

from models.mpoc.blockchain_network.simulation import Simulator
from config import *


def parse_arguments():
    parser = ArgumentParser()

    algorithms = [DevBBO, RW_GWO, L_SHADE, JSO,
                  OriginalABC, HI_WOA, DhoH, I_GWO,
                  EliteMultiGA, AugmentedAEO, DevSMA]
    
    parser.add_argument('-d', '--dim', type=int, default=10, help='number of dimensions')
    parser.add_argument('-a', '--algorithms', default=algorithms, help='list of test algorithms')
    parser.add_argument('-r', '--run', type=int, default=10, help='number of run times')
    parser.add_argument('-z', '--problem_size', type=int, default=20, help='size of problem')
    parser.add_argument('-e', '--epoch_bound', type=int, default=250, help='number of iterations')
    parser.add_argument('-p', '--pop-size', type=int, default=20, help='population size')
    parser.add_argument('-o', '--output', default='./output', help='output path')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='number of parallel processes')
    parser.add_argument('-m', '--multiprocessing', default=True, help='Run on multiprocessing')
    parser.add_argument('-t', '--time_bound', default=50, type=int, help='Time bound for trainning (s)')
    parser.add_argument('-v', '--verbose', default=False, help='log fitness by epoch')
    parser.add_argument('-n', '--n_trials', default=10, type=int, help='number of trials')
    parser.add_argument('-x', '--mode', default='epoch_bound', help='run with the time bound or epoch bound')
    parser.add_argument('-q', '--n_values', default='all', help='number of values')
    parser.add_argument('-s', '--num_sim', default=1, help='number of simulation for each solution')

    return parser.parse_args()

def mpoc_function(solution):
    '''
    Simulate blockchain network by parameters
    '''
    # scenario = [num_round, num_peer_on_network, num_leader_each_round, num_candidate_leader, num_peer_in_round_1]

    if args.n_values == 75:
        scenario = [[100, 200, 21, 20, 75]]
    elif args.n_values == 100:
        scenario = [[100, 200, 21, 20, 100]]
    elif args.n_values == 125:
        scenario = [[100, 200, 21, 20, 125]]
    elif args.n_values == 'all':
        scenario = [[50, 50, 21, 20, 30]]

    fitness = 0
    for _scenario in scenario:
        for _ in range(args.num_sim):
            simulation_result = 0
            while simulation_result == 0:
                try:
                    simulator = Simulator(solution, _scenario[0], _scenario[1], _scenario[2], _scenario[3], _scenario[4], 0.1)
                    simulation_result = simulator.simulate_mdpos()   
                except Exception as e:
                    pass
            fitness += simulation_result
    fitness /= (args.num_sim * len(scenario))
    return fitness

def save_history(history, algorithm_name, save_path):
    data = DataFrame(history.list_current_best_fit, columns=['fitness'])
    os.makedirs(save_path, exist_ok=True)
    data.to_csv(os.path.join(save_path, algorithm_name + '.csv'), index=False)
    solution = open(os.path.join(save_path, algorithm_name + "_best_solution.txt"), "w")
    solution.write("{}, {}\n".format(history.list_global_best[-1].solution.tolist(), sum(history.list_epoch_time)))

def process(function_name, algorithm, dim, _iter):
    lowerbound = [App.LOWERBOUND] * 10
    upperbound = [App.UPPERBOUND] * 10
    
    bounds = FloatVar((lowerbound,) * dim, (upperbound,) * dim)
    if args.mode == 'fes_bound':
        termination = Termination(max_fe=args.nfes, max_early_stop=15)
    elif args.mode == 'epoch_bound':
        termination = Termination(max_epoch=args.epoch_bound, max_early_stop=15)
    else:
        termination = Termination(max_time=args.time_bound, max_early_stop=15)
    
    problem_dict = {
        "bounds": bounds,
        "minmax": "min",
        "obj_func": mpoc_function
    }

    alg = algorithm(epoch=args.epoch_bound, pop_size=args.pop_size)
    alg.solve(problem_dict, termination=termination)
    save_path = os.path.join(args.output, args.mode, str(dim), function_name, str(_iter))
    save_history(alg.history, algorithm.__name__, save_path)

if __name__ == '__main__':
    args = parse_arguments()

    task_list = []
    for algorithm in args.algorithms:
        task_list.extend([('MPOC', algorithm, App.DIMENSIONS, iter)
                            for iter in range(args.n_trials)])

    if args.multiprocessing:
        Parallel(n_jobs=args.jobs)(delayed(process)(*params) \
                                   for params in task_list)
    else:
        for params in task_list:
            process(*params)

