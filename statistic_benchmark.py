import os
from joblib import Parallel, delayed
from argparse import ArgumentParser
import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
import seaborn as sns

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
from mealpy.bio_based.SMA import DevSMA
from config import *


def parse_arguments():
    parser = ArgumentParser()

    functions = ['f' + str(idx) for idx in range(1, 31)]
    algorithms = [DevBBO, RW_GWO, L_SHADE, JSO,
                  OriginalABC, HI_WOA, DhoH, I_GWO,
                  EliteMultiGA, AugmentedAEO, DevSMA]

    D = [10, 30, 50, 100]

    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-d', '--dim', type=int, default=D, help='number of dimensions')
    parser.add_argument('-a', '--algorithms', default=algorithms, help='list of test algorithms')
    parser.add_argument('-o', '--output', default='./output_new_2', help='output path')
    parser.add_argument('-i', '--fig_path', default='./figures_new', help='figures path')
    parser.add_argument('-x', '--mode', default='fes_bound', help='run with the time bound or epoch bound')
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

    for dim in args.dim:
        for function_name in args.functions:
            data = DataFrame([np.arange(1, 1001)], index=['ID']).T
            boxplot_rows = []
            best_result = []
            std = []

            for algorithm in args.algorithms:
                try:
                    save_path = os.path.join(args.output, args.mode, str(dim), function_name)
                    fitness = []
                    df_concat = None
                    for iter in range(0, args.n_trials):
                        df = read_csv(os.path.join(save_path, str(iter), algorithm.__name__ + '.csv'))
                        # df = df.apply(lambda x: np.log(x))
                        fit = df['fitness'].min()
                        fitness.append(fit)
                        boxplot_rows.append({'algorithm': algorithm().name, 'fitness': fit})
                        df.index.name = 'ID'
                        df.rename(columns={'fitness': 'fitness_{}'.format(iter)}, inplace=True)
                        if df_concat is None:
                            df_concat = df
                        else:
                            df_concat = pd.merge(df_concat, df, on='ID', how='outer')
                    data[algorithm().name] = df_concat.mean(axis=1, skipna=True)
                    data = data.iloc[:1000]
                    data[algorithm().name] = data[algorithm().name].fillna(data[algorithm().name].min())
                    best_result.append(np.mean(fitness))
                    std.append(np.std(fitness))
                except Exception as e:
                    print(e)
                    print('Error: ', algorithm().name, function_name, dim)
                    best_result.append(np.nan)
                    std.append(np.nan)
                    continue

            # ranking for each algorithm
            ranking[function_name] = ranking_values(best_result)
            final_result[function_name] = best_result
            std_result[function_name] = std
            
            # apply smoothing
            data = data.rolling(window=10).mean()
            data.set_index('ID', inplace=True)
            
            data.plot()
            plt.xlabel('epoch')
            plt.ylabel('Fitness (log)')
            # set log view
            plt.yscale('log')
            # save figure
            save_path = os.path.join(args.fig_path, args.mode, str(dim))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, function_name + '.pdf'))
            plt.savefig(os.path.join(save_path, function_name + '.png'))
            
            plt.clf()
            std_data = DataFrame(boxplot_rows)
            bp = sns.boxplot(data=std_data, x='algorithm', y='fitness')
            bp.set(xlabel=None)
            plt.savefig(os.path.join(save_path, 'std_' + function_name + '.pdf'))
            plt.savefig(os.path.join(save_path, 'std_' + function_name + '.png'))
            
            # plt.show()
            save_path = os.path.join(args.output, args.mode, str(dim), 'results')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            data.to_csv(os.path.join(save_path, function_name + '.csv'))

        ranking_df = DataFrame(ranking, index=[algorithm().name for algorithm in args.algorithms],
                               columns=args.functions)
        ranking_df.to_csv(os.path.join(args.output, args.mode, str(dim), 'ranking.csv'))

        final_result_df = DataFrame(final_result, index=[algorithm().name for algorithm in args.algorithms],
                                    columns=args.functions)
        final_result_df.to_csv(os.path.join(args.output, args.mode, str(dim), 'final_result.csv'))
        std_result_df = DataFrame(std_result, index=[algorithm().name for algorithm in args.algorithms],
                                  columns=args.functions)
        std_result_df.to_csv(os.path.join(args.output, args.mode, str(dim), 'std_result.csv'))
        