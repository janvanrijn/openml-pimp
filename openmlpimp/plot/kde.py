import arff
import argparse
import collections
import json
import matplotlib
import numpy as np
import openmlpimp
import os
import matplotlib.pyplot as plt

from scipy.stats import rv_discrete

from ConfigSpace.hyperparameters import CategoricalHyperparameter, NumericalHyperparameter


def parse_args():
    parser = argparse.ArgumentParser(description='Plot PDF diagrams according to KernelDensity Estimator')
    all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                       'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                       'qda', 'random_forest', 'sgd']
    all_classifiers = ['adaboost', 'random_forest']
    parser.add_argument('--flow_id', type=int, default=7707, help='the OpenML flow id')
    parser.add_argument('--classifier', type=str, default='libsvm_svc', help='the OpenML flow id')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the OpenML study id')
    parser.add_argument('--fixed_parameters', type=json.loads, default={'kernel': 'sigmoid'}, help='Will only use configurations that have these parameters fixed')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/cache_kde', help="Directory containing cache files")
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/pdf', help="Directory to save the result files")
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/nemo/experiments/priorbased_experiments', help="Adds samples obtained from a result directory")

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
                values[canonical_name].extend([arff_file['data'][x][idx] for x in range(len(arff_file['data']))])
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
        X_prime[value] += (1.0 / len(X))
    distrib = rv_discrete(values=(list(range(len(X_prime))), list(X_prime.values())))

    fig, ax = plt.subplots()
    # TODO: resampled from dist, but will do.
    ax.hist(distrib.rvs(size=100), range=(0, len(X_prime)))
    ax.legend(loc='upper left')

    plt.savefig(output_dir + parameter_name + '.png', bbox_inches='tight')
    plt.close()


def plot_numeric(hyperparameter, data, histo_keys, output_dir, parameter_name, resolution=100):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    factor = 1.0
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']
    min = np.power(hyperparameter.lower, factor)
    max = np.power(hyperparameter.upper, factor)
    if max < hyperparameter.upper:
        max = hyperparameter.upper * factor
    fig, axes = plt.subplots(1, figsize=(8, 6))

    for index, name in enumerate(data):
        if name in histo_keys:
            pass
            # axes[0].hist(data[name], resolution, normed=True, facecolor=colors[index], alpha=0.75)
        else:
            if hyperparameter.log:
                X_values_plot = np.logspace(np.log(min), np.log(max), resolution)
                axes.set_xscale("log")
            else:
                X_values_plot = np.linspace(min, max, resolution)

            # plot pdfs
            distribution = openmlpimp.utils.priors.gaussian_kde_wrapper(hyperparameter, hyperparameter.name, data[name])
            axes.plot(X_values_plot, distribution.pdf(X_values_plot), colors[index]+'-', lw=5, alpha=0.6, label=name.replace('_', ' '))

        # plot cdfs
        # sorted = np.sort(np.array(data[name]))
        # yvals = np.arange(1, len(sorted) + 1) / float(len(sorted))
        # axes[1].step(sorted, yvals, linewidth=1, c=colors[index], label=name)

    # add original data points
    #if 'gaussian_kde' in data:
    #    axes.plot(data['gaussian_kde'], -0.005 - 0.01 * np.random.random(len(data['gaussian_kde'])), '+k')

    # axis and labels
    #axes[1].legend(loc='upper left')
    axes.set_xlim(min, max)

    # plot
    plt.savefig(output_dir + parameter_name + '.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    tick_fontsize = 18
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['xtick.labelsize'] = tick_fontsize
    matplotlib.rcParams['ytick.labelsize'] = tick_fontsize

    args = parse_args()

    folder_suffix = openmlpimp.utils.fixed_parameters_to_suffix(args.fixed_parameters)
    output_dir = args.output_directory + '/hyperband_5/' + str(args.flow_id) + '/' + folder_suffix
    cache_dir = args.cache_directory + '/hyperband_5/' + str(args.flow_id) + '/' + folder_suffix
    results_dir = args.result_directory + '/hyperband_5/' + str(args.flow_id) + '/' + folder_suffix

    configspace = openmlpimp.utils.get_config_space_casualnames(args.classifier, args.fixed_parameters)

    obtained_results = {}
    if args.result_directory is not None:
        for strategy in os.listdir(results_dir):
            res = obtain_sampled_parameters(os.path.join(results_dir, strategy))
            if len(res):
                obtained_results[strategy] = res

    param_priors = openmlpimp.utils.obtain_priors(cache_dir, args.study_id, args.flow_id, configspace, args.fixed_parameters, holdout=None, bestN=10)

    for param_name, priors in param_priors.items():
        if all(x == priors[0] for x in priors):
            continue
        current_parameter = configspace.get_hyperparameter(param_name)
        histo_keys = set()
        if isinstance(current_parameter, NumericalHyperparameter):
            data = collections.OrderedDict({'gaussian_kde': priors})
            for strategy in obtained_results:
                strategy_name = openmlpimp.utils.plot._determine_name(strategy)
                data[strategy_name] = np.array(obtained_results[strategy][param_name], dtype=np.float64)
                histo_keys.add(strategy_name)
            plot_numeric(current_parameter, data, histo_keys, output_dir + '/', param_name)
        elif isinstance(current_parameter, CategoricalHyperparameter):
            plot_categorical(priors, output_dir + '/', param_name)
