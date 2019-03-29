import arff
import argparse
import fanova.fanova
import fanova.visualizer
import itertools
import json
import numpy as np
import logging
import openmlcontrib
import openmlpimp
import os
import pandas as pd
import sklearnbot


# to plot: <openml_pimp_root>/examples/plot/plot_fanova_aggregates.py
def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../../hypeCNN/data/12param/fanova-resnet.arff', type=str)
    parser.add_argument('--output_directory', default=os.path.expanduser('~/experiments/openml-pimp'), type=str)
    parser.add_argument('--classifier', default='resnet', type=str)
    parser.add_argument('--config_library', default='openmlpimp', type=str)
    parser.add_argument('--measure', default='predictive_accuracy', type=str)
    # parser.add_argument('--plot_marginals', action='store_true', default=True)
    # parser.add_argument('--plot_extension', default='pdf', type=str)
    # parser.add_argument('--plot_resolution', default=100, type=int)
    parser.add_argument('--comb_size', default=2, type=int)
    parser.add_argument('--n_trees', default=16, type=int)
    # parser.add_argument('--resolution', default=100, type=int)
    parser.add_argument('--task_id', default=None, type=str)
    parser.add_argument('--task_id_column', default='dataset', type=str)
    parser.add_argument('--subsample', default=None, type=int)
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
    logging.info('Start %s: %s' % (os.path.basename(__file__), vars(args)))
    
    with open(args.dataset_path, 'r') as fp:
        arff_dataset = arff.load(fp)
    if args.config_library == 'sklearnbot':
        config_space = sklearnbot.config_spaces.get_config_space(args.classifier, None)
    elif args.config_library == 'openmlpimp':
        config_space = openmlpimp.configspaces.get_config_space(args.classifier, None)
    else:
        raise ValueError('Could not identify config library: %s' % args.config_library)

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
    task_ids = data[args.task_id_column].unique()
    if args.task_id:
        task_ids = [args.task_id]

    result = list()
    for t_idx, task_id in enumerate(task_ids):
        logging.info('Running fanova on task %s (%d/%d)' % (task_id, t_idx + 1, len(task_ids)))
        data_task = data[data[args.task_id_column] == task_id]
        del data_task[args.task_id_column]
        # now dataset is gone, and all categoricals are converted, we can convert to float
        data_task = data_task.astype(np.float)
        if args.subsample:
            indices = np.random.choice(len(data_task), args.subsample, replace=False)
            data_task = data_task.iloc[indices]
        logging.info('Dimensions: %s (out of (%s)) %s' % (str(data_task.shape),
                                                          str(data.shape),
                                                          '[Subsampled]' if args.subsample else ''))
        assert len(data_task) >= min(100, args.subsample if args.subsample is not None else 100)
        os.makedirs(args.output_directory, exist_ok=True)
        X_data = data_task[config_space.get_hyperparameter_names()].values
        y_data = data_task[args.measure].values
        evaluator = fanova.fanova.fANOVA(X=X_data,
                                         Y=y_data,
                                         config_space=config_space,
                                         n_trees=args.n_trees)
        vis = fanova.visualizer.Visualizer(evaluator,
                                           config_space,
                                           args.output_directory,
                                           y_label='Predictive Accuracy')
        indices = list(range(len(config_space.get_hyperparameters())))
        for comb_size in range(1, args.comb_size + 1):
            for h_idx in itertools.combinations(indices, comb_size):
                param_names = np.array(config_space.get_hyperparameter_names())[np.array(h_idx)]
                logging.info('-- Calculating marginal for %s' % param_names)
                importance = evaluator.quantify_importance(h_idx)[h_idx]
                if comb_size == 1:
                    visualizer_res = vis.generate_marginal(h_idx[0], args.resolution)
                    # visualizer returns mean, std and potentially grid
                    avg_marginal = np.array(visualizer_res[0])
                elif comb_size == 2:
                    visualizer_res = vis.generate_pairwise_marginal(h_idx, args.resolution)
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
    result_path = os.path.join(args.output_directory, 'fanova_%s_depth_%d%s.csv' % (
        args.classifier, args.comb_size, '_%s' % args.task_id if args.task_id else ''))
    df_result.to_csv(result_path)
    logging.info('resulting csv: %s' % result_path)
    logging.info('To plot, run <openml_pimp_root>/examples/plot/plot_fanova_aggregates.py')


if __name__ == '__main__':
    run(read_cmd())
