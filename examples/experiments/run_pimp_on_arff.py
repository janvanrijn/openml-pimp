import arff
import argparse
import ConfigSpace
import fanova.fanova
import fanova.visualizer
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
import openmlcontrib
import openmlpimp
import os
import pandas as pd
import sklearnbot
import typing


# to plot: <openml_pimp_root>/examples/plot/plot_fanova.py
def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../../hypeCNN/data/12param/fanova-resnet.arff', type=str)
    parser.add_argument('--output_directory', default=os.path.expanduser('~/experiments/openml-pimp'), type=str)
    parser.add_argument('--classifier', default='resnet', type=str)
    parser.add_argument('--config_library', default='openmlpimp', type=str)
    parser.add_argument('--measure', default='predictive_accuracy', type=str)
    parser.add_argument('--plot_marginals', action='store_true', default=True)
    parser.add_argument('--comb_size', default=2, type=int)
    parser.add_argument('--n_trees', default=16, type=int)
    parser.add_argument('--resolution', default=100, type=int)
    parser.add_argument('--task_id', default=None, type=str)
    parser.add_argument('--task_id_column', default='dataset', type=str)
    parser.add_argument('--subsample', default=None, type=int)
    args_, misc = parser.parse_known_args()

    return args_


def plot_single_marginal(config_space: ConfigSpace.ConfigurationSpace,
                         name_prefix: str,
                         hyperparameter_idx: int,
                         visualizer: fanova.visualizer.Visualizer,
                         directory: str,
                         y_range: typing.Optional[typing.Tuple[int, int]]):
    plt.close('all')
    plt.clf()
    hyperparameter = config_space.get_hyperparameter_by_idx(hyperparameter_idx)
    os.makedirs(directory, exist_ok=True)
    outfile_name = os.path.join(directory, '%s__%s.png' % (name_prefix, hyperparameter.replace(os.sep, "_")))
    visualizer.plot_marginal(hyperparameter_idx, show=False)

    x1, x2, _, _ = plt.axis()
    if y_range:
        plt.axis((x1, x2, y_range[0], y_range[1]))
    plt.savefig(outfile_name)
    logging.info('saved marginal to: %s' % outfile_name)


def plot_pairwise_marginal(config_space: ConfigSpace.ConfigurationSpace,
                           name_prefix: str,
                           hyperparameter_idx: typing.Tuple[int],
                           visualizer: fanova.visualizer.Visualizer,
                           directory: str,
                           z_range: typing.Optional[typing.Tuple[int, int]]):
    plt.close('all')
    plt.clf()
    if len(hyperparameter_idx) != 2:
        raise ValueError()
    indices = [hyperparameter_idx, (hyperparameter_idx[1], hyperparameter_idx[0])]
    for hp1_hp2 in indices:
        hp1 = config_space.get_hyperparameter_by_idx(hp1_hp2[0])
        hp2 = config_space.get_hyperparameter_by_idx(hp1_hp2[1])
        os.makedirs(directory, exist_ok=True)
        outfile_name = os.path.join(directory, '%s__%s__%s.png' % (name_prefix,
                                                                   hp1.replace(os.sep, "_"),
                                                                   hp2.replace(os.sep, "_")))
        try:
            visualizer.plot_pairwise_marginal(hp1_hp2, resolution=100, show=False)

            ax = plt.gca()
            if z_range:
                ax.set_zlim3d(z_range[0], z_range[1])
            plt.savefig(outfile_name)
            logging.info('saved marginal to: %s' % outfile_name)
        except IndexError as e:
            logging.warning('IndexError with hyperparameters %s and %s: %s' % (hp1, hp2, e))


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
        if args.subsample:
            indices = np.random.choice(len(data_task), args.subsample, replace=False)
            data_task = data_task.iloc[indices]
        logging.info('Dimensions: %s (out of (%s)) %s' % (str(data_task.shape),
                                                          str(data.shape),
                                                          '[Subsampled]' if args.subsample else ''))
        assert len(data_task) > 100
        
        evaluator = fanova.fanova.fANOVA(X=data_task[config_space.get_hyperparameter_names()].values,
                                         Y=data_task[args.measure].values,
                                         config_space=config_space,
                                         n_trees=args.n_trees)

        os.makedirs(args.output_directory, exist_ok=True)

        vis = fanova.visualizer.Visualizer(evaluator,
                                           config_space,
                                           args.output_directory, y_label='Predictive Accuracy')
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

                    if args.plot_marginals:
                        plot_single_marginal(
                            config_space, task_id, h_idx[0], vis,
                            os.path.join(args.output_directory, str(task_id), 'singular'),
                            None
                        )
                elif comb_size == 2:
                    visualizer_res = vis.generate_pairwise_marginal(h_idx, args.resolution)
                    # visualizer returns grid names and values
                    avg_marginal = np.array(visualizer_res[1])

                    if args.plot_marginals:
                        plot_pairwise_marginal(
                            config_space, task_id, h_idx, vis,
                            os.path.join(args.output_directory, str(task_id), 'pairwise'),
                            None
                        )

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
    logging.info('To plot, run <openml_pimp_root>/examples/plot/plot_fanova.py')


if __name__ == '__main__':
    run(read_cmd())
