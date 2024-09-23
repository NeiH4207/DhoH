import itertools
import os
from joblib import Parallel, delayed
from argparse import ArgumentParser
from pandas.core.frame import DataFrame

from models.algorithms.DhoH import DhoH
from models.benchmark.cec2017 import Functions

from mealpy import FloatVar
from mealpy.utils.termination import Termination
from config import *


def parse_arguments():
    parser = ArgumentParser()

    functions = ['f' + str(idx) for idx in range(1, 31)]
    
    D = 10

    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-d', '--dim', type=int, default=D, help='number of dimensions')
    parser.add_argument('-e', '--epoch_bound', type=int, default=250, help='number of iterations')
    parser.add_argument('-p', '--pop-size', type=int, default=100, help='population size')
    parser.add_argument('-o', '--output', default='./output_tuning', help='output path')
    parser.add_argument('-j', '--jobs', type=int, default=64, help='number of parallel processes')
    parser.add_argument('-m', '--multiprocessing', default=True, help='Run on multiprocessing')
    parser.add_argument('-t', '--time_bound', default=200, type=int, help='Time bound for trainning (s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-n', '--n_trials', default=10, type=int, help='number of trials')
    parser.add_argument('-x', '--mode', default='fes_bound',
                        help='run with the time boun, epoch bound or fes bound')
    parser.add_argument('-c', '--nfes', type=int, default=1000000, help='number of iterations')

    return parser.parse_args()

def save_history(history, algorithm_name, save_path):
    data = DataFrame(history.list_current_best_fit, columns=['fitness'])
    os.makedirs(save_path, exist_ok=True)
    data.to_csv(os.path.join(save_path, algorithm_name + '.csv'), index=False)
    solution = open(os.path.join(save_path, algorithm_name + "_best_solution.txt"), "w")
    solution.write("{}, {}\n".format(history.list_global_best[-1].solution.tolist(), sum(history.list_epoch_time)))

def process(function_name, dim, alpha, drop_rate, r1, r2, _iter):
    log_file = open("xlogs.txt", "a")
    log = f'Running on function = {function_name}(alpha={alpha}, r1={r1}, r2={r2}), drop_rate={drop_rate}, D = {dim}, trial {_iter}'
    log_file.write(log + '\n')
    log_file.close()
    
    alg_name = DhoH.__name__ + f'_alpha_{alpha}_drop_rate_{drop_rate}_r1_{r1}_r2_{r2}'
    
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

    function = Functions(function_name, dim)

    alg = DhoH(
        epoch=args.epoch_bound, 
        pop_size=args.pop_size,
        alpha=alpha,
        drop_rate=drop_rate,
        r1=r1,
        r2=r2,
    )

    alg.solve(problem_dict, termination=termination)
    
    
    save_path = os.path.join(args.output, args.mode, str(dim), function_name, str(_iter))
    save_history(alg.history, alg_name, save_path)


if __name__ == '__main__':
    args = parse_arguments()

    param_grid = {
        'alpha': [0.2, 0.3, 0.4, 0.5],
        'drop_rate': [0.0, 0.2, 0.4, 0.6],
        'r1': [1.5],
        'r2': [1.25],
    }

    task_list = []

    instances = list(itertools.product(param_grid['alpha'], param_grid['drop_rate'], param_grid['r1'], param_grid['r2']))
    for function_name in args.functions:
        for alpha, drop_rate, r1, r2 in instances:
            task_list.extend([(function_name, args.dim, alpha, drop_rate, r1, r2, iter)
                              for iter in range(args.n_trials)])

    if args.multiprocessing:
        Parallel(n_jobs=args.jobs)(delayed(process)(*params) \
                                   for params in task_list)
    else:
        for params in task_list:
            process(*params)

