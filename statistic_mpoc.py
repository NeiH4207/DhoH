import os
from joblib import Parallel, delayed
from argparse import ArgumentParser
import numpy as np

from pandas.core.frame import DataFrame
from pandas.io.parsers import read_csv

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
from models.benchmark.functions import Functions
from config import *
from matplotlib import pyplot as plt


def parse_arguments():
    parser = ArgumentParser()

    algorithms = [HI_WOA, DHO, ChaoticWhale, ImprovedGA,
                  OriginalGSKA, OriginalHGS, L_SHADE, 
                  BaseSMA, OriginalSMA, AdaptiveEO]
    algorithms = [HI_WOA, DHO, ImprovedGA, OriginalGSKA, L_SHADE, AdaptiveEO]
    parser.add_argument('-a', '--algorithms', default=algorithms, help='list of test algorithms')
    parser.add_argument('-o', '--output', default='./output', help='output path')
    parser.add_argument('-i', '--fig_path', default='./figures', help='figures path')
    parser.add_argument('-x', '--mode', default='epoch_bound', help='run with the time bound or epoch bound')
    parser.add_argument('-n', '--n_trials', default=10, type=int, help='number of trials')
    
    return parser.parse_args()

def ranking_values(values):
    id = [i for i in range(len(values))]
    id = sorted(id, key=lambda x: values[x])
    ranking = [-1] * len(values)
    ranking[id[0]] = 1
    curr_rank = 1
    cur_value = values[id[0]]
    for i in id[1:]:
        if values[i] == cur_value:
            ranking[i] = curr_rank
        else:
            curr_rank += 1
            ranking[i] = curr_rank
            cur_value = values[i]
    return ranking
    
if __name__ == '__main__':
    args = parse_arguments()
    
    ranking = {}
    final_result = {}
    std_result = {}
    
    data = DataFrame()
    best_result = []
    std = []
    name = 'MPOC'
    
    for algorithm in args.algorithms:
        alg_name = algorithm(None).name        
        save_path = os.path.join(args.output, args.mode, name)
        fitness = []
        df = read_csv(os.path.join(save_path, str(6), algorithm.__name__ + '.csv'))
        for iter in range(0, args.n_trials):
            fitness.append(df['fitness'].min())
        data[algorithm(None).name] = df
        # fill nan values with min value
        data[alg_name] = data[alg_name].fillna(data[alg_name].min())
        best_result.append(np.mean(fitness))
        std.append(np.std(fitness))
            
        # ranking for each algorithm
    ranking[name] = ranking_values(best_result)
    final_result[name] = best_result
    std_result[name] = std
    data.plot()
    plt.xlabel('epoch')
    plt.ylabel('Fitness')
    # save figure
    save_path = os.path.join(args.fig_path, args.mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, name + '.pdf'))
    plt.savefig(os.path.join(save_path, name + '.png'))
    # plt.show()
    save_path = os.path.join(args.output, args.mode, 'results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data.to_csv(os.path.join(save_path, name + '.csv'))
        
    ranking_df = DataFrame(ranking, index=[algorithm(None).name for algorithm in args.algorithms],
                           columns=[name])
    ranking_df.to_csv(os.path.join(args.output, args.mode, 'ranking.csv'))
    
    final_result_df = DataFrame(final_result, index=[algorithm(None).name for algorithm in args.algorithms],
                            columns=[name])         
    final_result_df.to_csv(os.path.join(args.output, args.mode, 'final_result.csv'))
    std_result_df = DataFrame(std_result, index=[algorithm(None).name for algorithm in args.algorithms],
                            columns=[name])
    std_result_df.to_csv(os.path.join(args.output, args.mode, 'std_result.csv'))