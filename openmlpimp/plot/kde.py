import arff
import argparse
import collections
import copy
import json
import numpy as np
import openmlpimp
import os
import matplotlib.pyplot as plt

from scipy.stats import rv_discrete

import autosklearn.constants
from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter, NumericalHyperparameter
from autosklearn.util.pipeline import get_configuration_space


def parse_args():
    parser = argparse.ArgumentParser(description='Plot PDF diagrams according to KernelDensity Estimator')
    all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                       'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                       'qda', 'random_forest', 'sgd']
    all_classifiers = ['adaboost', 'random_forest']
    parser.add_argument('--flow_id', type=int, default=6970, help='the OpenML flow id')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the OpenML study id')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='adaboost', help='the classifier to execute')
    parser.add_argument('--fixed_parameters', type=json.loads, default=None,
                        help='Will only use configurations that have these parameters fixed')
    parser.add_argument('--bestN', type=int, default=10, help='number of best setups to consider for creating the priors')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/cache_kde', help="Directory containing cache files")
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/pdf', help="Directory to save the result files")
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/nemo/experiments/random_search_prior', help="Adds samples obtained from a result directory")

    args = parser.parse_args()
    return args


def obtain_sampled_parameters(directory):
    import glob
    files = glob.glob(directory + '/*/*.arff')
    values = collections.defaultdict(list)
    for file in files:
        with open(file, 'r') as fp:
            arff_file = arff.load(fp)
        for idx, attribute in enumerate(arff_file['attributes']):
            attribute_name = attribute[0]
            if attribute_name.startswith('parameter_'):
                canonical_name = attribute_name.split('__')[-1]
                values[canonical_name].extend([arff_file['data'][x][idx] for x in range(len((arff_file['data'])))])
    return values


def plot_categorical(X, output_dir, parameter_name):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    X_prime = collections.OrderedDict()
    for value in X:
        if value not in X_prime:
            X_prime[value] = 0
        X_prime[value] = X_prime[value] + (1.0 / len(X))
    distrib = rv_discrete(values=(list(range(len(X_prime))), list(X_prime.values())))

    fig, ax = plt.subplots()
    # TODO: resampled from dist, but will do.
    ax.hist(distrib.rvs(size=100), range=(0, len(X_prime)))
    ax.legend(loc='upper left')

    plt.savefig(output_dir + parameter_name + '.png', bbox_inches='tight')
    plt.close()


def plot_numeric(hyperparameter, distributions, output_dir, parameter_name, data=None):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    lines = ['r-', 'b-', 'g-']
    X_values_plot = np.linspace(hyperparameter.lower, hyperparameter.upper, 1000)
    fig, ax = plt.subplots()

    for index, name in enumerate(distributions):
        distribution = distributions[name]
        ax.plot(X_values_plot, distribution.pdf(X_values_plot), lines[index], lw=5, alpha=0.6, label=name)

    ax.legend(loc='upper left')

    if data is not None:
        ax.plot(data, -0.005 - 0.01 * np.random.random(data.shape[0]), '+k')

    ax.set_xlim(hyperparameter.lower, hyperparameter.upper)
    if hyperparameter.log:
        plt.xscale("log", log=2)
    plt.savefig(output_dir + parameter_name + '.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = parse_args()

    cache_folder_suffix = openmlpimp.utils.fixed_parameters_to_suffix(args.fixed_parameters)
    important_parameters = copy.deepcopy(args.fixed_parameters) if args.fixed_parameters is not None else {}
    important_parameters['bestN'] = args.bestN
    save_folder_suffix = openmlpimp.utils.fixed_parameters_to_suffix(important_parameters)

    output_dir = args.output_directory + '/' + args.classifier + '/' + save_folder_suffix
    cache_dir = args.cache_directory + '/' + args.classifier + '/' + cache_folder_suffix
    results_dir = args.result_directory + '/' + args.classifier + '/' + save_folder_suffix

    configuration_space = get_configuration_space(
        info={'task': autosklearn.constants.MULTICLASS_CLASSIFICATION, 'is_sparse': 0},
        include_estimators=[args.classifier],
        include_preprocessors=['no_preprocessing'])

    hyperparameters = openmlpimp.utils.configspace_to_relevantparams(configuration_space)
    obtained_results = {}
    if args.result_directory is not None:
        for strategy in os.listdir(results_dir):
            obtained_results[strategy] = obtain_sampled_parameters(os.path.join(results_dir, strategy))

    X = openmlpimp.utils.obtain_priors(cache_dir, args.study_id, args.flow_id, hyperparameters, args.fixed_parameters, holdout=None, bestN=10)
    prior_param_grid = openmlpimp.utils.get_prior_paramgrid(cache_dir, args.study_id, args.flow_id, hyperparameters, args.fixed_parameters)

    for param_name, priors in prior_param_grid.items():
        current_parameter = hyperparameters[param_name]
        if isinstance(current_parameter, NumericalHyperparameter):
            distributions = collections.OrderedDict({'gaussian_kde': prior_param_grid[param_name]})
            for strategy in obtained_results:
                data = np.array(obtained_results[strategy][param_name], dtype=np.float64)
                distributions[strategy] = openmlpimp.utils.priors.gaussian_kde_wrapper(current_parameter, current_parameter.name, data)
            plot_numeric(current_parameter, distributions, output_dir + '/', param_name, X[param_name])
        elif isinstance(current_parameter, CategoricalHyperparameter):
            plot_categorical(X[param_name], output_dir + '/', param_name)
