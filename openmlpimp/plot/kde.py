import argparse
import json
import numpy as np
import openmlpimp
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import autosklearn.constants
from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter, NumericalHyperparameter
from autosklearn.util.pipeline import get_configuration_space


def parse_args():
    parser = argparse.ArgumentParser(description='Plot PDF diagrams according to KernelDensity Estimator')
    all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                       'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                       'qda', 'random_forest', 'sgd']
    all_classifiers = ['adaboost', 'random_forest']
    parser.add_argument('--flow_id', type=int, default=6952, help='the OpenML flow id')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the OpenML study id')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='libsvm_svc', help='the classifier to execute')
    parser.add_argument('--fixed_parameters', type=json.loads, default={'kernel': 'sigmoid'},
                        help='Will only use configurations that have these parameters fixed')

    args = parser.parse_args()
    return args


def plot(X, output_dir, parameter_name, log_scale, min_value, max_value):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    X_plot = np.linspace(min_value, max_value, 1000)
    fig, ax = plt.subplots()

    if log_scale:
        kde = gaussian_kde(np.log2(X))
        log_dens = kde.pdf(np.log2(X_plot))
    else:
        kde = gaussian_kde(X)
        log_dens = kde.pdf(X_plot)

    ax.plot(X_plot, log_dens, 'r-', lw = 5, alpha = 0.6, label='gaussian kde')

    ax.legend(loc='upper left')
    ax.plot(X, -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

    ax.set_xlim(min_value, max_value)
    if log_scale:
        plt.xscale("log", log=2)
    plt.savefig(output_dir + parameter_name + '.png', bbox_inches='tight')


if __name__ == '__main__':
    args = parse_args()
    cache_dir = '/home/vanrijn/experiments/cache_kde'
    output_dir = '/home/vanrijn/experiments/pdf'
    if args.fixed_parameters:
        save_folder_suffix = [param + '_' + value for param, value in args.fixed_parameters.items()]
        save_folder_suffix = '/' + '__'.join(save_folder_suffix)
    else:
        save_folder_suffix = '/vanilla'
    output_dir = output_dir + '/' + args.classifier + '/' + save_folder_suffix
    cache_dir = cache_dir + '/' + args.classifier + '/' + save_folder_suffix

    configuration_space = get_configuration_space(
        info={'task': autosklearn.constants.MULTICLASS_CLASSIFICATION, 'is_sparse': 0},
        include_estimators=[args.classifier],
        include_preprocessors=['no_preprocessing'])

    hyperparameters = {}
    for parameter in configuration_space.get_hyperparameters():
        splittedname = parameter.name.split(':')
        if splittedname[0] not in ['classifier', 'imputation']:
            continue
        if isinstance(parameter, Constant):
            continue
        if isinstance(parameter, CategoricalHyperparameter):
            if len(parameter.choices) <= 1:
                continue
            continue # TEMP
        hyperparameters[splittedname[-1]] = parameter

        if parameter.name == 'classifier:random_forest:max_features':
            hyperparameters[splittedname[-1]].lower = 0.1
            hyperparameters[splittedname[-1]].upper = 0.9

    print(hyperparameters)

    X = openmlpimp.utils.obtain_priors(cache_dir, args.study_id, args.flow_id, hyperparameters, args.fixed_parameters)
    
    #----------------------------------------------------------------------
    # Plot a 1D density example
    for param_name, parameter in hyperparameters.items():
        logscale = False
        if isinstance(parameter, NumericalHyperparameter):
            logscale = parameter.log
        try:
            plot(X[param_name], output_dir + '/', param_name, logscale , parameter.lower, parameter.upper)
        except np.linalg.linalg.LinAlgError as e:
            sys.stderr.write(param_name + ': ' + str(e))
