import arff
import argparse
import fanova
import itertools
import json
import numpy as np
import logging
import openmlcontrib
import sklearnbot


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../KDD2018/data/arff/adaboost.arff', type=str)
    parser.add_argument('--measure', default='predictive_accuracy', type=str)
    parser.add_argument('--classifier', default='adaboost', type=str)
    parser.add_argument('--n_trees', default=16, type=int)
    parser.add_argument('--comb_size', default=1, type=int)
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
        indices = list(range(len(config_space.get_hyperparameters())))
        for comb_size in range(1, args.comb_size + 1):
            for idx in itertools.combinations(indices, comb_size):
                param_names = np.array(config_space.get_hyperparameter_names())[idx]
                logging.info('-- Calculating marginal for %s' % param_names)
                importance = evaluator.quantify_importance(idx)[idx]
                print(importance)
                current = {
                
                }
                
                result.append(current)


if __name__ == '__main__':
    run(read_cmd())
