import os
from joblib import Parallel, delayed
from argparse import ArgumentParser
import numpy as np
import pandas as pd

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
# from models.algorithms.DHO_Exp import DHO_ABr1, DHO_ABr2, DHO_NS10, DHO_NS15, DHO_NS20, DHO
from models.algorithms.C_WOA import ChaoticWhale
from models.benchmark.functions import Functions
from config import *
from matplotlib import pyplot as plt


def parse_arguments():
    parser = ArgumentParser()

    functions = ['uni1', 'uni2', 'uni3', 'uni4', 'uni5',
                 'multi1', 'multi2', 'multi3', 'multi4', 'multi5',
                 'hybrid1', 'hybrid2', 'hybrid3', 'hybrid4', 'hybrid5',
                 'composition1', 'composition2', 'composition3', 'composition4', 'composition5']
    
    algorithms = [HI_WOA, ImprovedGA,
                  OriginalGSKA, OriginalHGS, L_SHADE, 
                  BaseSMA, OriginalSMA, AdaptiveEO, DHO]
    # functions = ['uni4', 'multi4', 'composition4', 'composition5']
    
    # algorithms = [DHO_NS10, DHO_NS15, DHO_NS20]

    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-a', '--algorithms', default=algorithms, help='list of test algorithms')
    parser.add_argument('-o', '--output', default='./output', help='output path')
    parser.add_argument('-i', '--fig_path', default='./figures', help='figures path')
    parser.add_argument('-x', '--mode', default='time_bound', help='run with the time bound or epoch bound')
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
    
    for function_name in args.functions:
        data = DataFrame()
        best_result = []
        std = []
        
        for algorithm in args.algorithms:
            save_path = os.path.join(args.output, args.mode, function_name)
            fitness = []
            df_concat = None
            for iter in range(0, args.n_trials):
                df = read_csv(os.path.join(save_path, str(iter), algorithm.__name__ + '.csv'))
                # df = df.apply(lambda x: np.log(x))
                fitness.append(df['fitness'].min())
                df.index.name = 'ID'
                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.merge(df_concat, df, on='ID')
            data[algorithm().name] = df_concat.mean(axis=1)
            # set limit rows (500)
            data = data.iloc[:500]
            # fill nan values with min value
            data[algorithm().name] = data[algorithm().name].fillna(data[algorithm().name].min())
            best_result.append(np.mean(fitness))
            std.append(np.std(fitness))
            
        # ranking for each algorithm
        ranking[function_name] = ranking_values(best_result)
        final_result[function_name] = best_result
        std_result[function_name] = std
        data.plot()
        plt.xlabel('epoch')
        plt.ylabel('Fitness (log)')
        # set log view
        plt.yscale('log')
        # save figure
        save_path = os.path.join(args.fig_path, args.mode)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, function_name + '.pdf'))
        plt.savefig(os.path.join(save_path, function_name + '.png'))
        # plt.show()
        save_path = os.path.join(args.output, args.mode, 'results')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data.to_csv(os.path.join(save_path, function_name + '.csv'))
        
    ranking_df = DataFrame(ranking, index=[algorithm().name for algorithm in args.algorithms],
                           columns=args.functions)
    ranking_df.to_csv(os.path.join(args.output, args.mode, 'ranking.csv'))
    
    final_result_df = DataFrame(final_result, index=[algorithm().name for algorithm in args.algorithms],
                            columns=args.functions)         
    final_result_df.to_csv(os.path.join(args.output, args.mode, 'final_result.csv'))
    std_result_df = DataFrame(std_result, index=[algorithm().name for algorithm in args.algorithms],
                            columns=args.functions)
    std_result_df.to_csv(os.path.join(args.output, args.mode, 'std_result.csv'))