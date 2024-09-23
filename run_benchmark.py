import os
from joblib import Parallel, delayed
from argparse import ArgumentParser
from pandas.core.frame import DataFrame

from models.algorithms.DhoH import DhoH
from models.benchmark.cec2017 import Functions
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
from config import *


def parse_arguments():
    parser = ArgumentParser()

    functions = ['f' + str(idx) for idx in range(1, 31)]
    
    algorithms = [DevBBO, RW_GWO, L_SHADE, JSO,
                  OriginalABC, HI_WOA, DhoH, I_GWO,
                  EliteMultiGA, AugmentedAEO, DevSMA]
    
    algorithms = [JSO, I_GWO]
    
    D = [10, 30, 50, 100]

    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-d', '--dim', type=int, default=D, help='number of dimensions')
    parser.add_argument('-a', '--algorithms', default=algorithms, help='list of test algorithms')
    parser.add_argument('-e', '--epoch_bound', type=int, default=250, help='number of iterations')
    parser.add_argument('-c', '--nfes', type=int, default=1000000, help='number of iterations')
    parser.add_argument('-p', '--pop-size', type=int, default=100, help='population size')
    parser.add_argument('-o', '--output', default='./output_new_2', help='output path')
    parser.add_argument('-j', '--jobs', type=int, default=40, help='number of parallel processes')
    parser.add_argument('-m', '--multiprocessing', default=True, help='Run on multiprocessing')
    parser.add_argument('-t', '--time_bound', default=200, type=int, help='Time bound for trainning (s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-n', '--n_trials', default=10, type=int, help='number of trials')
    parser.add_argument('-x', '--mode', default='fes_bound',
                        help='run with the time bound, epoch bound or fes bound')

    return parser.parse_args()


def save_history(history, algorithm_name, save_path):
    data = DataFrame(history.list_current_best_fit, columns=['fitness'])
    os.makedirs(save_path, exist_ok=True)
    data.to_csv(os.path.join(save_path, algorithm_name + '.csv'), index=False)
    solution = open(os.path.join(save_path, algorithm_name + "_best_solution.txt"), "w")
    solution.write("{}, {}\n".format(history.list_global_best[-1].solution.tolist(), sum(history.list_epoch_time)))

def process(function_name, algorithm, dim, _iter):
    function = Functions(function_name, dim)
    bounds = FloatVar((function.lb,) * function.D, (function.ub,) * function.D)
    if args.mode == 'fes_bound':
        termination = Termination(max_fe=args.nfes, max_early_stop=15)
    elif args.mode == 'epoch_bound':
        termination = Termination(max_epoch=args.epoch_bound, max_early_stop=15)
    else:
        termination = Termination(max_time=args.time_bound, max_early_stop=15)
    
    problem_dict = {
        "bounds": bounds,
        "minmax": "min",
        "obj_func": function.value
    }

    alg = algorithm(epoch=args.epoch_bound, pop_size=args.pop_size)
    alg.solve(problem_dict, termination=termination)
    save_path = os.path.join(args.output, args.mode, str(dim), function_name, str(_iter))
    save_history(alg.history, algorithm.__name__, save_path)

if __name__ == '__main__':
    args = parse_arguments()

    task_list = []
    for dim in args.dim:
        for function_name in args.functions:
            for algorithm in args.algorithms:
                task_list.extend([(function_name, algorithm, dim, iter)
                                  for iter in range(args.n_trials)])

    if args.multiprocessing:
        Parallel(n_jobs=args.jobs)(delayed(process)(*params) \
                                   for params in task_list)
    else:
        for params in task_list:
            process(*params)

