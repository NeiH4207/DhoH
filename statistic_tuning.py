import os
from joblib import Parallel, delayed
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import itertools

from pandas.core.frame import DataFrame
from pandas.io.parsers import read_csv

from config import *
from matplotlib import pyplot as plt
import seaborn as sns

from models.algorithms.DhoH import DhoH


def parse_arguments():
    parser = ArgumentParser()

    functions = ['f' + str(idx) for idx in range(1, 31)]
    # functions = ['uni4', 'multi4', 'composition4', 'composition5']

    # algorithms = [DHO_NS10, DHO_NS15, DHO_NS20]
    D = 10

    parser.add_argument('-f', '--functions', default=functions, help='list of benchmark functions')
    parser.add_argument('-d', '--dim', type=int, default=D, help='number of dimensions')
    parser.add_argument('-o', '--output', default='./output_tuning', help='output path')
    parser.add_argument('-i', '--fig_path', default='./figures_tuning', help='figures path')
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
    
    param_grid = {
        'alpha': [0.2, 0.3, 0.4, 0.5],
        'drop_rate': [0.0, 0.2, 0.4, 0.6],
        'r1': [1.5],
        'r2': [1.25],
    }

    ranking = {}
    final_result = {}
    std_result = {}

    instances = list(itertools.product(param_grid['alpha'], param_grid['drop_rate'], param_grid['r1'], param_grid['r2']))
    dim = args.dim
    for function_name in args.functions:
        data = DataFrame([np.arange(1, 201)], index=['ID']).T
        boxplot_rows = []
        best_result = []
        std = []


        for alpha, drop_rate, r1, r2 in instances:
            save_path = os.path.join(args.output, args.mode, str(dim), function_name)
            name = f'[{alpha} - {drop_rate} - {r1} - {r2}]'
            fitness = []
            df_concat = None
            for iter in range(0, args.n_trials):
                try:
                    df = read_csv(os.path.join(save_path, str(iter), DhoH.__name__ + f'_alpha_{alpha}_drop_rate_{drop_rate}_r1_{r1}_r2_{r2}.csv'))
                    # df = df.apply(lambda x: np.log(x))
                    fit = df['fitness'].min()
                    fitness.append(fit)
                    boxplot_rows.append({'algorithm': name, 'fitness': fit})
                    df.index.name = 'ID'
                    df.rename(columns={'fitness': 'fitness_{}'.format(iter)}, inplace=True)
                    if df_concat is None:
                        df_concat = df
                    else:
                        df_concat = pd.merge(df_concat, df, on='ID', how='outer')
                except Exception as e:
                    print(e)
                    print('Error: ', name, function_name, dim)
            
            try:
                data[name] = df_concat.mean(axis=1, skipna=True)
                data = data.iloc[:1000]
                data[name] = data[name].fillna(data[name].min())
                best_result.append(np.mean(fitness))
                std.append(np.std(fitness))
            except Exception as e:
                print(e)
                print('Error: ', DhoH.__name__, function_name, dim)
                continue

        if len(best_result) == 0:
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
        # plot horizontal boxplot
        # set figure size
        plt.figure(figsize=(13, 7))
        bp = sns.boxplot(data=std_data, x='fitness', y='algorithm', showfliers=False)
        bp.set(xlabel=None)
        plt.savefig(os.path.join(save_path, 'std_' + function_name + '.pdf'))
        plt.savefig(os.path.join(save_path, 'std_' + function_name + '.png'))
        
        
        # plt.show()
        save_path = os.path.join(args.output, args.mode, str(dim), 'results')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data.to_csv(os.path.join(save_path, function_name + '.csv'))

    prams_str = []
    for alpha, drop_rate, r1, r2 in instances:
        prams_str.append(f'[{alpha} - {drop_rate} - {r1} - {r2}]')
        
    # remove functions that have enough len(params) < len(functions)
    
    for func in args.functions:
        if func not in final_result:
            continue
        if len(final_result[func]) < len(prams_str):
            for key in [ranking, final_result, std_result]:
                key.pop(func)

    final_result_df = DataFrame(final_result, index=[param for param in prams_str],
                                columns=args.functions)
    
    final_result_df.to_csv(os.path.join(args.output, args.mode, str(dim), 'final_result.csv'))
        
    ranking_df = DataFrame(ranking, index=[param for param in prams_str],
                            columns=args.functions)
    ranking_df.to_csv(os.path.join(args.output, args.mode, str(dim), 'ranking.csv'))
    
    std_result_df = DataFrame(std_result, index=[param for param in prams_str],
                                columns=args.functions)
    std_result_df.to_csv(os.path.join(args.output, args.mode, str(dim), 'std_result.csv'))
    