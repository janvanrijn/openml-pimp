import arff
import argparse
import fanova
from fanova.visualizer import Visualizer
import itertools
import json
import numpy as np
import logging
import openmlcontrib
import os
import pandas as pd
import sklearnbot


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../KDD2018/data/arff/adaboost.arff', type=str)
    parser.add_argument('--output_directory', default=os.path.expanduser('~/experiments/openml-pimp'), type=str)
    parser.add_argument('--classifier', default='adaboost', type=str)
    parser.add_argument('--measure', default='predictive_accuracy', type=str)
    parser.add_argument('--comb_size', default=1, type=int)
    parser.add_argument('--n_trees', default=16, type=int)
    parser.add_argument('--resolution', default=100, type=int)
    args_, misc = parser.parse_known_args()

    return args_


def get_dataset_metadata(dataset_path):
    with open(dataset_path) as fp:
        first_line = fp.readline()
        if first_line[0] != '%':
            raise ValueError('arff data file should start with comment for meta-data')
    meta_data = json.loads(first_line[1:])
    return meta_data


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    with open(args.dataset_path, 'r') as fp:
        arff_dataset = arff.load(fp)
    config_space = sklearnbot.config_spaces.get_config_space(args.classifier, None)
    data = openmlcontrib.meta.arff_to_dataframe(arff_dataset, config_space)
    data = openmlcontrib.meta.integer_encode_dataframe(data, config_space)
    meta_data = get_dataset_metadata(args.dataset_path)
    if args.measure not in data.columns.values:
        raise ValueError('Could not find measure in dataset: %s' % args.measure)
    if set(config_space.get_hyperparameter_names()) != set(meta_data['col_parameters']):
        missing_cs = set(meta_data['col_parameters']) - set(config_space.get_hyperparameter_names())
        missing_ds = set(config_space.get_hyperparameter_names()) - set(meta_data['col_parameters'])
        raise ValueError('ConfigSpace and hyperparameters of dataset do not '
                         'align. ConfigSpace misses: %s, dataset misses: %s' % (missing_cs, missing_ds))
    task_ids = data['task_id'].unique()

    result = list()
    for idx, task_id in enumerate(task_ids):
        logging.info('Running fanova on task %d (%d/%d)' % (task_id, idx + 1, len(task_ids)))
        data_task = data[data['task_id'] == task_id]
        
        evaluator = fanova.fanova.fANOVA(X=data_task[config_space.get_hyperparameter_names()].values,
                                         Y=data_task[args.measure].values,
                                         config_space=config_space,
                                         n_trees=args.n_trees)

        os.makedirs(args.output_directory, exist_ok=True)

        vis = Visualizer(evaluator, config_space, args.output_directory, y_label='Predictive Accuracy')
        indices = list(range(len(config_space.get_hyperparameters())))
        for comb_size in range(1, args.comb_size + 1):
            for idx in itertools.combinations(indices, comb_size):
                param_names = np.array(config_space.get_hyperparameter_names())[np.array(idx)]
                logging.info('-- Calculating marginal for %s' % param_names)
                importance = evaluator.quantify_importance(idx)[idx]
                if comb_size == 1:
                    visualizer_res = vis.generate_marginal(idx[0], args.resolution)
                    # visualizer returns mean, std and potentially grid
                    avg_marginal = np.array(visualizer_res[0])
                elif comb_size == 2:
                    visualizer_res = vis.generate_pairwise_marginal(idx, args.resolution)
                    # visualizer returns grid names and values
                    avg_marginal = np.array(visualizer_res[1])
                else:
                    raise ValueError('No support yet for higher dimensions than 2. Got: %d' % comb_size)
                difference_max_min = max(avg_marginal.reshape((-1,))) - min(avg_marginal.reshape((-1,)))

                current = {
                    'task_id': task_id,
                    'hyperparameter': ' / '.join(param_names),
                    'n_hyperparameters': len(param_names),
                    'importance_variance': importance['individual importance'],
                    'importance_max_min': difference_max_min,
                }
                
                result.append(current)
    df_result = pd.DataFrame(result)
    df_result.to_csv(os.path.join(args.output_directory, 'fanova_%s_depth_%d.csv' % (args.classifier, args.comb_size)))


if __name__ == '__main__':
    run(read_cmd())
