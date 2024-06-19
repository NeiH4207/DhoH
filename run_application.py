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
from models.algorithms.C_WOA import ChaoticWhale
from models.mpoc.blockchain_network.simulation import Simulator
from config import *


def parse_arguments():
    parser = ArgumentParser()

    algorithms = [HI_WOA, DHO,
                  OriginalGSKA, OriginalHGS, L_SHADE, 
                  BaseSMA, AdaptiveEO]
                #  OriginalGSKA, OriginalHGS, L_SHADE, BaseSMA, OriginalSMA, AdaptiveEO]
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
                except Exception as ex:
                    pass
            fitness += simulation_result
    fitness /= (args.num_sim * len(scenario))
    return fitness

def run(solver, algorithm_name, _iter):
    print("Run {}".format(algorithm_name))
    best_sol, runtime, loss_train = solver.train()
    data = DataFrame(loss_train, columns=['fitness'])
    save_path = os.path.join(args.output, args.mode, 'MPOC', str(_iter))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data.to_csv(os.path.join(save_path, algorithm_name + '.csv'), index=False)
    solution = open(os.path.join(save_path, algorithm_name + "_final_solution_MPOC.txt"), "w")
    solution.write("{}, {}\n".format(best_sol[0].tolist(), runtime))
    log_file = open("logs.txt", "a")
    log_file.write('Finished ' + algorithm_name  + ': ' + str(best_sol[1]) + '\n')
    log_file.write("Total excution time: {}s\n".format(round(runtime, 3)) + '\n')
    print('Finished ' + algorithm_name  + ': ' , best_sol[1])
    print("Total excution time: {}s\n".format(round(runtime, 3)))
    log_file.close()

def process(_iter):
    print("Trial-", _iter)
        
    log_file = open("logs.txt", "a")
    log_file.close()
    function = mpoc_function
    lowerbound = [App.LOWERBOUND] * 10
    upperbound = [App.UPPERBOUND] * 10
    
    for algorithm in args.algorithms:
        alg = algorithm(function, lb=lowerbound, ub=upperbound,
            epoch=args.epoch_bound, pop_size=args.pop_size, mode=args.mode,
            time_bound=args.time_bound, verbose=args.verbose)
        run(alg, algorithm.__name__, _iter)

if __name__ == '__main__':
    args = parse_arguments()
                
    if args.multiprocessing:
        Parallel(n_jobs=args.jobs)(delayed(process)(iter) for iter in range(args.n_trials))
    else:
        for iter in range(args.n_trials):
            process(iter)
        
