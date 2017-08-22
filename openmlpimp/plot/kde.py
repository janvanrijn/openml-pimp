import argparse
import json
import numpy as np
import openml
import openmlpimp
import os
import pickle
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


def cache_priors(cache_directory, study_id, fixed_parameters):
    study = openml.study.get_study(study_id, 'tasks')
    if fixed_parameters is not None and len(fixed_parameters) > 0:
        setups = openmlpimp.utils.obtain_all_setups(flow=args.flow_id)

    best_setupids = {}
    for task_id in study.tasks:
        print("task", task_id)
        runs = openml.evaluations.list_evaluations("predictive_accuracy", task=[task_id], flow=[args.flow_id])
        best_score = 0.0
        for run in runs.values():
            score = run.value
            if run.setup_id not in setups:
                raise ValueError()
            if score > best_score and len(fixed_parameters) == 0:
                best_score = score
                best_setupids[task_id] = run.setup_id
            elif score > best_score and openmlpimp.utils.setup_complies_to_fixed_parameters(setups[run.setup_id],
                                                                                            'parameter_name',
                                                                                            fixed_parameters):
                best_score = score
                best_setupids[task_id] = run.setup_id

                # if len(best_setupids) > 10: break
    try:
        os.makedirs(cache_directory)
    except FileExistsError:
        pass

    with open(cache_directory + '/best_setup_per_task.pkl', 'wb') as f:
        pickle.dump(best_setupids, f, pickle.HIGHEST_PROTOCOL)


def obtain_priors(cache_directory, study_id, hyperparameters, fixed_parameters, holdout=None):
    filename = cache_directory + '/best_setup_per_task.pkl'
    if not os.path.isfile(filename):
        cache_priors(cache_directory, study_id, fixed_parameters)

    with open(filename, 'rb') as f:
        best_setupids = pickle.load(f)

    X = {parameter: list() for parameter in hyperparameters.keys()}
    setups = openml.setups.list_setups(setup=list(best_setupids.values()), flow=args.flow_id)

    for task_id, setup_id in best_setupids.items():
        if task_id == holdout:
            print('Holdout task %d' %task_id)
            continue
        paramname_paramidx = {param.parameter_name: idx for idx, param in setups[setup_id].parameters.items()}
        for parameter in hyperparameters.keys():
            param = setups[setup_id].parameters[paramname_paramidx[parameter]]
            X[parameter].append(float(param.value))

    for parameter in X:
        X[parameter] = np.array(X[parameter])
    return X


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

    X = obtain_priors(cache_dir, args.study_id, hyperparameters, args.fixed_parameters)

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