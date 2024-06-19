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
from models.algorithms.I_GWO import I_GWO
from models.algorithms.DHO import DHO
from models.algorithms.JSO import JSO
# from models.algorithms.DHO_Exp import DHO_ABr1, DHO_ABr2, DHO_NS10, DHO_NS15, DHO_NS20, DHO_old
from models.algorithms.C_WOA import ChaoticWhale
from models.benchmark.cec2017 import Functions
from config import *


def parse_arguments():
    parser = ArgumentParser()

    functions = ['f' + str(idx) for idx in range(20, 30)]

#     algorithms = [HI_WOA, ImprovedGA,
#                   OriginalGSKA, OriginalHGS, L_SHADE,
#                   BaseSMA, OriginalSMA, AdaptiveEO, I_GWO, DHO]
    algorithms = [DHO]
#     algorithms = [I_GWO]
#     D = [10, 30, 50, 100]
    D = [10]

    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-d', '--dim', type=int, default=D, help='number of dimensions')
    parser.add_argument('-a', '--algorithms', default=algorithms, help='list of test algorithms')
    parser.add_argument('-e', '--epoch_bound', type=int, default=250, help='number of iterations')
    parser.add_argument('-p', '--pop-size', type=int, default=100, help='population size')
    parser.add_argument('-o', '--output', default='./output_new', help='output path')
    parser.add_argument('-j', '--jobs', type=int, default=80, help='number of parallel processes')
    parser.add_argument('-m', '--multiprocessing', default=False, help='Run on multiprocessing')
    parser.add_argument('-t', '--time_bound', default=200, type=int, help='Time bound for trainning (s)')
    parser.add_argument('-v', '--verbose', action='store_true', default=True, help='verbose mode')
    parser.add_argument('-n', '--n_trials', default=1, type=int, help='number of trials')
    parser.add_argument('-x', '--mode', default='time_bound',
                        help='run with the time boun, epoch bound or fes bound')

    return parser.parse_args()


def run(solver, algorithm_name, save_path):
    best_sol, runtime, loss_train = solver.train()
    data = DataFrame(loss_train, columns=['fitness'])
    os.makedirs(save_path, exist_ok=True)
    data.to_csv(os.path.join(save_path, algorithm_name + '.csv'), index=False)
    solution = open(os.path.join(save_path, algorithm_name + "_final_solution.txt"), "w")
    solution.write("{}, {}\n".format(best_sol[0].tolist(), runtime))

    log_file = open("logs.txt", "a")
    log_file.write('Finished ' + algorithm_name + ': ' + str(best_sol[1]) + '\n')
    log_file.write("Total excution time: {}s\n".format(round(runtime, 3)) + '\n')
    print('Finished ' + algorithm_name + ': ', best_sol[1])
    print("Total excution time: {}s\n".format(round(runtime, 3)))
    log_file.close()


def process(function_name, algorithm, dim, _iter):
    log_file = open("xlogs.txt", "a")
    log = f'Running on function = {function_name}, alggorithm = {algorithm.__name__}, D = {dim}, trial {_iter}'
    log_file.write(log + '\n')
    log_file.close()
    print(log)

    function = Functions(function_name, dim)
    lowerbound = [function.lb] * function.D
    upperbound = [function.ub] * function.D

    alg = algorithm(function.value, lb=lowerbound, ub=upperbound,
                    epoch=args.epoch_bound, pop_size=args.pop_size, mode=args.mode,
                    time_bound=args.time_bound, fes_bound=100000 * dim, verbose=args.verbose)

    save_path = os.path.join(args.output, args.mode, str(dim), function_name, str(_iter))
    run(alg, algorithm.__name__, save_path)


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

