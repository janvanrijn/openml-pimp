import argparse
import collections
import csv
import json
import matplotlib.pyplot as plt
import openmlpimp
import os
from statistics import median


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/experiments/fanova/triples/rbf', help='the directory to load the experiments from')

    args = parser.parse_args()
    return args


def _format(name):
    mapping_plain = {
        'strategy': 'imputation',
        'max_features': 'max. features',
        'min_samples_leaf': 'min. samples leaf',
        'min_samples_split': 'min. samples split',
        'criterion': 'split criterion',
        'learning_rate': 'learning rate',
        'max_depth': 'max. depth',
        'n_estimators': 'iterations',
        'algorithm': 'algorithm',
    }
    mapping_short = {
        'strategy': 'imputation',
        'max_features': 'max. feat.',
        'min_samples_leaf': 'samples leaf',
        'min_samples_split': 'samples split',
        'criterion': 'split criterion',
        'learning_rate': 'learning r.',
        'max_depth': 'max. depth',
        'n_estimators': 'iterations',
        'algorithm': 'algo.',
    }

    parts = name.split('__')
    for idx, part in enumerate(parts):
        if part in mapping_plain:
            if len(parts) < 3:
                parts[idx] = mapping_plain[part]
            else:
                parts[idx] = mapping_short[part]

    return ' / '.join(parts)


def to_ranks_file(marginal_contribution, all_tasks):
    with open(args.result_directory + '/ranks_all.csv', 'w') as csvfile:
        fieldnames = ['task_id']
        fieldnames.extend([_format(key) for key in marginal_contribution.keys()])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for idx, task_id in enumerate(all_tasks):
            current_point = {_format(hyperparameter): marginal_contribution[hyperparameter][idx] for hyperparameter in marginal_contribution.keys()}
            current_point['task_id'] = 'Task %s' %task_id
            writer.writerow(current_point)


def to_ranks_plain_file(sorted_values, keys, tasks_order):
    with open(args.result_directory + '/ranks_plain_all.csv', 'w') as csvfile:
        fieldnames = ['task_id', 'param_id', 'param_name', 'variance_contribution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task_idx, task_id in enumerate(tasks_order):
            for hp_idx, hyperparameter in enumerate(keys):
                current_point = {'task_id': task_id, 'param_id': _format(hyperparameter), 'param_name': _format(hyperparameter), 'variance_contribution': sorted_values[hp_idx][task_idx]}
                writer.writerow(current_point)


def marginal_plots(sorted_values, keys):
    plt.figure()
    plt.violinplot(list(sorted_values), list(range(len(sorted_values))))
    plt.plot([-0.5, len(sorted_values) - 0.5], [0, 0], 'k-', linestyle='--', lw=1)
    plt.xticks(list(range(len(sorted_values))), list(keys), rotation=45, ha='right')
    output_file = args.result_directory + '/marginal_contributions.pdf'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def determine_relevant(data, max_items=None, max_interactions=None):
    sorted_values = []
    keys = []
    interactions_seen = 0
    for key in sorted(data, key=lambda k: median(data[k]), reverse=True):
        if '__' in key:
            interactions_seen += 1
            if interactions_seen > max_interactions:
                continue

        sorted_values.append(data[key])
        keys.append(key)

    if max_items is not None:
        sorted_values = sorted_values[:max_items]
        keys = keys[:max_items]

    return sorted_values, keys


if __name__ == '__main__':
    args = parse_args()
    total_ranks, marginal_contribution = openmlpimp.utils.obtain_marginal_contributions(args.result_directory)
    sorted_values, keys = determine_relevant(marginal_contribution, max_interactions=3)

