import argparse
import openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

import autosklearn.constants
from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter
from autosklearn.util.pipeline import get_configuration_space


def parse_args():
    parser = argparse.ArgumentParser(description='Plot PDF diagrams according to KernelDensity Estimator')
    all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                       'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                       'qda', 'random_forest', 'sgd']
    all_classifiers = ['adaboost', 'random_forest']
    parser.add_argument('--flow_id', type=int, default=6969, help='the OpenML flow id')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='random_forest', help='the classifier to execute')

    args = parser.parse_args()
    return args


def plot(X, output_dir, parameter_name, min_value, max_value):
    X_plot = np.linspace(min_value, max_value, (max_value - min_value) * 100)[:, np.newaxis]
    fig, ax = plt.subplots()

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-', label="kernel = '{0}'".format('gaussian'))

    ax.legend(loc='upper left')
    ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

    ax.set_xlim(min_value, max_value)
    plt.savefig(output_dir + parameter_name + '.png', bbox_inches='tight')


if __name__ == '__main__':
    args = parse_args()

    output_dir = '/home/vanrijn/experiments/pdf/'

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

    X = {parameter: list() for parameter in hyperparameters.keys()}

    study = openml.study.get_study('OpenML100', 'tasks')
    best_setupids = {}
    for task_id in study.tasks:
        print("task", task_id)
        runs = openml.evaluations.list_evaluations("predictive_accuracy", task=[task_id], flow=[args.flow_id])
        best_score = 0.0
        for run in runs.values():
            score = run.value
            if score > best_score:
                best_score = score
                best_setupids[task_id] = run.setup_id
        # if len(best_setupids) > 10: break

    setups = openml.setups.list_setups(setup=list(best_setupids.values()), flow=args.flow_id)
    for setup_id in best_setupids.values():
        paramname_paramidx = {param.parameter_name: idx for idx, param in setups[setup_id].parameters.items()}
        for parameter in hyperparameters.keys():
            param = setups[setup_id].parameters[paramname_paramidx[parameter]]
            X[parameter].append([float(param.value)])

    for parameter in X:
        X[parameter] = np.array(X[parameter])

    #----------------------------------------------------------------------
    # Plot a 1D density example
    for param_name, parameter in hyperparameters.items():
        plot(X[param_name], output_dir + args.classifier + '/', param_name, parameter.lower, parameter.upper)
