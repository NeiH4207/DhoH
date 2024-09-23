from joblib import Parallel, delayed
from argparse import ArgumentParser

from joblib import Parallel, delayed
from pandas.core.frame import DataFrame
import os
import sys
sys.path.append(os.getcwd())

from config import *
from models.algorithms.DHO import DHO
from models.algorithms.EO import AdaptiveEO
from models.algorithms.GSKA import OriginalGSKA
from models.algorithms.HGS import OriginalHGS
from models.algorithms.HI_WOA import HI_WOA
from models.algorithms.LSHADE import L_SHADE
from models.algorithms.RCGA_rdn import ImprovedGA
from models.algorithms.SMA import BaseSMA, OriginalSMA
from models.algorithms.I_GWO import I_GWO
from models.benchmark.cec2011.cec2011_class import *


def parse_arguments():
    parser = ArgumentParser()

    functions = [Problem01, Problem02, Problem03, Problem04, Problem05, Problem06, Problem07,
                 Problem09, Problem10, Problem11_1, Problem11_2, Problem11_3, Problem11_4, Problem11_5,
                 Problem11_6, Problem11_8, Problem11_9, Problem11_10, Problem12, Problem13]
    algorithms = [HI_WOA, ImprovedGA,
                  OriginalGSKA, OriginalHGS, L_SHADE,
                  BaseSMA, AdaptiveEO, I_GWO, DHO]

    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-a', '--algorithms', default=algorithms, help='list of test algorithms')
    parser.add_argument('-e', '--epoch_bound', type=int, default=250, help='number of iterations')
    parser.add_argument('-p', '--pop-size', type=int, default=100, help='population size')
    parser.add_argument('-o', '--output', default='./output_rw', help='output path')
    parser.add_argument('-j', '--jobs', type=int, default=80, help='number of parallel processes')
    parser.add_argument('-m', '--multiprocessing', default=True, help='Run on multiprocessing')
    parser.add_argument('-t', '--time_bound', default=200, type=int, help='Time bound for trainning (s)')
    parser.add_argument('-fes', '--fes_bound', default=150000, type=int,
                        help='Function evaluation bound for trainning (s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-n', '--n_trials', default=3, type=int, help='number of trials')
    parser.add_argument('-x', '--mode', default='time_bound',
                        help='run with the time boun, epoch bound or fes bound')
    parser.add_argument('-w', '--write', dest='write', action='store_true', 
                        help='write (overwrite) or append results')

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


def process(func, algorithm, _iter):
    function = func()
    function_name = func.__name__
    
    save_path = os.path.join(args.output, args.mode, function_name, str(_iter))
    if not args.write and os.path.exists(os.path.join(save_path, algorithm.__name__ + '.csv')):
        return
        
    
    log_file = open("xlogs.txt", "a")
    log = f'Running on function = {function_name}, algorithm = {algorithm.__name__}, trial {_iter}'
    log_file.write(log + '\n')
    log_file.close()
    print(log)

    lowerbound = list(function.lb)
    upperbound = list(function.ub)

    alg = algorithm(function.evaluate, lb=lowerbound, ub=upperbound,
                    epoch=args.epoch_bound, pop_size=args.pop_size, mode=args.mode,
                    time_bound=args.time_bound, fes_bound=args.fes_bound, verbose=args.verbose)
    
    run(alg, algorithm.__name__, save_path)


if __name__ == '__main__':
    args = parse_arguments()

    task_list = []
    for iter in range(args.n_trials):
        for function in args.functions:
            for algorithm in args.algorithms:
                task_list.append((function, algorithm, iter))

    if args.multiprocessing:
        Parallel(n_jobs=args.jobs, backend='threading')(delayed(process)(*params)
                                                              for params in task_list)
    else:
        for params in task_list:
            process(*params)

