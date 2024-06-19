import itertools
import os
from joblib import Parallel, delayed
from argparse import ArgumentParser

from pandas.core.frame import DataFrame

from models.algorithms.HI_WOA import HI_WOA
from models.algorithms.DMeSR_PSO import MentoringSwarm
from models.algorithms.C_PSO import ChaosSwarm
from models.algorithms.RCGA_rdn import ImprovedGA
from models.algorithms.GSKA import OriginalGSKA
from models.algorithms.HGS import OriginalHGS
from models.algorithms.LSHADE import L_SHADE
from models.algorithms.SMA import BaseSMA, OriginalSMA
from models.algorithms.EO import AdaptiveEO
from models.algorithms.DHO import DHO
# from models.algorithms.DHO_Exp import DHO_ABr1, DHO_ABr2, DHO_NS10, DHO_NS15, DHO_NS20, DHO
from models.algorithms.C_WOA import ChaoticWhale
from models.benchmark.cec2017 import Functions
from config import *


def parse_arguments():
    parser = ArgumentParser()

    functions = ['f' + str(idx) for idx in range(1, 31)]

    algorithms = [HI_WOA, ImprovedGA,
                  OriginalGSKA, OriginalHGS, L_SHADE,
                  BaseSMA, OriginalSMA, AdaptiveEO, DHO]
    # algorithms = [DHO]
    # algorithms = [L_SHADE]
    # D = [10, 30, 50, 100]
    D = 10

    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-d', '--dim', type=int, default=D, help='number of dimensions')
    parser.add_argument('-e', '--epoch_bound', type=int, default=250, help='number of iterations')
    parser.add_argument('-p', '--pop-size', type=int, default=100, help='population size')
    parser.add_argument('-o', '--output', default='./output_tuning', help='output path')
    parser.add_argument('-j', '--jobs', type=int, default=80, help='number of parallel processes')
    parser.add_argument('-m', '--multiprocessing', default=True, help='Run on multiprocessing')
    parser.add_argument('-t', '--time_bound', default=200, type=int, help='Time bound for trainning (s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-n', '--n_trials', default=10, type=int, help='number of trials')
    parser.add_argument('-x', '--mode', default='time_bound',
                        help='run with the time boun, epoch bound or fes bound')

    return parser.parse_args()


def process(function_name, dim, n_search, r1, r2, _iter):
    log_file = open("xlogs.txt", "a")
    log = f'Running on function = {function_name}(n_search={n_search}, r1={r1}, r2={r2}), D = {dim}, trial {_iter}'
    log_file.write(log + '\n')
    log_file.close()
    print(log)

    function = Functions(function_name, dim)
    lowerbound = [function.lb] * function.D
    upperbound = [function.ub] * function.D

    solver = DHO(function.value, lb=lowerbound, ub=upperbound,
                 epoch=args.epoch_bound, pop_size=args.pop_size, mode=args.mode,
                 time_bound=args.time_bound, fes_bound=10000 * dim, verbose=args.verbose,
                 n_searchs=n_search, r1=r1, r2=r2)

    save_path = os.path.join(args.output, args.mode, str(dim), function_name, str(_iter))
    file_name = f'n_search_{n_search}_r1_{r1}_r2_{r2}'

    best_sol, runtime, loss_train = solver.train()
    data = DataFrame(loss_train, columns=['fitness'])
    os.makedirs(save_path, exist_ok=True)
    data.to_csv(os.path.join(save_path, file_name + '.csv'), index=False)
    solution = open(os.path.join(save_path, file_name + "_final_solution.txt"), "w")
    solution.write("{}, {}\n".format(best_sol[0].tolist(), runtime))

    log_file = open("logs.txt", "a")
    log_file.write('Finished ' + file_name + ': ' + str(best_sol[1]) + '\n')
    log_file.write("Total excution time: {}s\n".format(round(runtime, 3)) + '\n')
    print('Finished ' + file_name + ': ', best_sol[1])
    print("Total excution time: {}s\n".format(round(runtime, 3)))
    log_file.close()


if __name__ == '__main__':
    args = parse_arguments()

    param_grid = {
        "n_searchs": [11, 13],
        'r1': [1.05, 1.1, 1.15, 1.2, 1.25],
        'r2': [1.05, 1.1, 1.15, 1.2, 1.25],
    }

    task_list = []

    instances = list(itertools.product(param_grid['n_searchs'], param_grid['r1'], param_grid['r2']))
    for function_name in args.functions:
        for n_search, r1, r2 in instances:
            task_list.extend([(function_name, args.dim, n_search, r1, r2, iter)
                              for iter in range(args.n_trials)])

    if args.multiprocessing:
        Parallel(n_jobs=args.jobs)(delayed(process)(*params) \
                                   for params in task_list)
    else:
        for params in task_list:
            process(*params)

