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
# from models.algorithms.DHO import DHO
from models.algorithms.DHO_Exp import DHO_ABr1, DHO_ABr2, DHO_NS10, DHO_NS15, DHO_NS20, DHO
from models.algorithms.C_WOA import ChaoticWhale
from models.benchmark.functions import Functions
from config import *


def parse_arguments():
    parser = ArgumentParser()

    functions = ['uni1', 'uni2', 'uni3', 'uni4', 'uni5',
                 'multi1', 'multi2', 'multi3', 'multi4', 'multi5',
                 'hybrid1', 'hybrid2', 'hybrid3', 'hybrid4', 'hybrid5',
                 'composition1', 'composition2', 'composition3', 'composition4', 'composition5']

    functions = ['uni1', 'uni2', 'uni3', 'uni4', 'uni5',
                 'multi1', 'multi2', 'multi3', 'multi4', 'multi5']
    
    algorithms = [HI_WOA, ImprovedGA,
                  OriginalGSKA, OriginalHGS, L_SHADE, 
                  BaseSMA, OriginalSMA, AdaptiveEO, DHO]
    # algorithms = [DHO]
    # algorithms = [L_SHADE]
    
    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-d', '--dim', type=int, default=10, help='number of dimensions')
    parser.add_argument('-a', '--algorithms', default=algorithms, help='list of test algorithms')
    parser.add_argument('-r', '--run', type=int, default=50, help='number of run times')
    parser.add_argument('-z', '--problem_size', type=int, default=50, help='size of problem')
    parser.add_argument('-e', '--epoch_bound', type=int, default=250, help='number of iterations')
    parser.add_argument('-p', '--pop-size', type=int, default=100, help='population size')
    parser.add_argument('-o', '--output', default='./output', help='output path')
    parser.add_argument('-j', '--jobs', type=int, default=10, help='number of parallel processes')
    parser.add_argument('-m', '--multiprocessing', default=True, help='Run on multiprocessing')
    parser.add_argument('-t', '--time_bound', default=200, type=int, help='Time bound for trainning (s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-n', '--n_trials', default=10, type=int, help='number of trials')
    parser.add_argument('-x', '--mode', default='time_bound', help='run with the time bound or epoch bound')

    return parser.parse_args()


def run(solver, function_name, algorithm_name, _iter):
    best_sol, runtime, loss_train = solver.train()
    data = DataFrame(loss_train, columns=['fitness'])
    save_path = os.path.join(args.output, args.mode, function_name, str(_iter))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data.to_csv(os.path.join(save_path, algorithm_name + '.csv'), index=False)
    solution = open(os.path.join(save_path, algorithm_name + "_final_solution.txt"), "w")
    solution.write("{}, {}\n".format(best_sol[0].tolist(), runtime))
    
    log_file = open("logs.txt", "a")
    log_file.write('Finished ' + algorithm_name  + ': ' + str(best_sol[1]) + '\n')
    log_file.write("Total excution time: {}s\n".format(round(runtime, 3)) + '\n')
    print('Finished ' + algorithm_name  + ': ' , best_sol[1])
    print("Total excution time: {}s\n".format(round(runtime, 3)))
    log_file.close()

def process(_iter):
    print("Trial-", _iter)
        
    for function_name in args.functions:
        log_file = open("xlogs.txt", "a")
        log_file.write('Running on function ' + function_name + '\n')
        log_file.close()
        print('Running on function ' + function_name)
        function = Functions(function_name, args.dim)
        solvers = []
        lowerbound = [Benchmark.LOWERBOUND] * function.D
        upperbound = [Benchmark.UPPERBOUND] * function.D
        
        for algorithm in args.algorithms:
            alg = algorithm(function.value, lb=lowerbound, ub=upperbound,
                epoch=args.epoch_bound, pop_size=args.pop_size, mode=args.mode,
                time_bound=args.time_bound, verbose=args.verbose)
            run(alg, function_name, algorithm.__name__, _iter)

if __name__ == '__main__':
    args = parse_arguments()
                
    if args.multiprocessing:
        Parallel(n_jobs=args.jobs)(delayed(process)(iter) \
                for iter in range(args.n_trials))
    else:
        for iter in range(len(args.n_trials)):
            process(iter)
        
