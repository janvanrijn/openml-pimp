import argparse
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import logging
import openml
import os
import pandas as pd
import seaborn as sns
import typing


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fanova_result_file',
                        default=os.path.expanduser('~/experiments/cnn_fanova/fanova_resnet_depth_2.csv'),
                        type=str)
    parser.add_argument('--output_directory', default=os.path.expanduser('~/experiments/openml-pimp'), type=str)
    parser.add_argument('--measure', default='predictive_accuracy', type=str)
    parser.add_argument('--n_combi_params', default=3, type=int)
    args_, misc = parser.parse_known_args()

    return args_


def calculate_cutoff_value(medians: pd.DataFrame, column_name: str, n_combi_params: typing.Optional[int]):
    medians_sorted = medians[medians['n_hyperparameters'] > 1].sort_values(column_name)
    cutoff = 0.0
    if n_combi_params is not None and len(medians_sorted) > n_combi_params:
        cutoff = medians_sorted[column_name][-1 * n_combi_params]
    return cutoff


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    df = pd.read_csv(args.fanova_result_file)
    medians = df.groupby('hyperparameter')['n_hyperparameters', 'importance_variance', 'importance_max_min'].median()
    df = df.join(medians, on='hyperparameter', how='left', rsuffix='_median')

    # vanilla boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    cutoff_value_variance = calculate_cutoff_value(medians, 'importance_variance', args.n_combi_params) - 0.000001
    sns.boxplot(x='hyperparameter', y='importance_variance',
                data=df.query('n_hyperparameters == 1 or importance_variance_median >= %f' % cutoff_value_variance).sort_values('importance_variance_median'), ax=ax1)
    ax1.set_title('fanova (variance)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    cutoff_value_max_min = calculate_cutoff_value(medians, 'importance_max_min', args.n_combi_params) - 0.000001
    sns.boxplot(x='hyperparameter', y='importance_max_min',
                data=df.query('n_hyperparameters == 1 or importance_max_min_median >= %f' % cutoff_value_max_min).sort_values('importance_max_min_median'), ax=ax2)
    ax2.set_title('fanova (max-min)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    os.makedirs(args.output_directory, exist_ok=True)
    output_file = os.path.join(args.output_directory, '%s.png' % os.path.basename(args.fanova_result_file))
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)

    # best per data feature
    #qualities = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_id=df['task_id'].unique()), orient='index')
    #qualities = qualities[['tid', 'NumberOfInstances', 'NumberOfFeatures']]


if __name__ == '__main__':
    run(read_cmd())
