import argparse
import arff
import copy
import collections
import json
import os
import openml
import openmlpimp

# Mounting CMD: sshfs fr_jv1031@login1.nemo.uni-freiburg.de:/home/fr/fr_fr/fr_jv1031 nemo/

def parse_args():
    all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest', 'sgd']
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--virtual_env', type=str, default=os.path.expanduser('~') + '/projects/pythonvirtual/plot2/bin/python', help='python virtual env for plotting')
    parser.add_argument('--scripts_dir', type=str, default=os.path.expanduser('~') + '/projects/plotting_scripts/scripts', help='directory to Katha\'s plotting scripts')
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/nemo/experiments/random_search_prior/', help='the directory to load the experiments from')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/optimizers/priors/', help='the directory to load the experiments from')

    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='adaboost', help='the classifier to execute')
    parser.add_argument('--bestN', type=int, default=10, help='number of best setups to consider for creating the priors')
    parser.add_argument('--fixed_parameters', type=json.loads, default=None, help='Will only use configurations that have these parameters fixed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    important_parameters = copy.deepcopy(args.fixed_parameters) if args.fixed_parameters is not None else {}
    important_parameters['bestN'] = args.bestN

    results_suffix = openmlpimp.utils.fixed_parameters_to_suffix(important_parameters)

    result_directory = args.result_directory + args.classifier + '/' + results_suffix + '/'
    output_directory = args.output_directory + args.classifier + '/' + results_suffix + '/'

    all_taskids = set()
    all_traces = collections.defaultdict(dict)

    all_strategies = os.listdir(result_directory)
    strategy_directories = {}
    for strategy in all_strategies:
        directory = os.path.join(result_directory, strategy)
        strategy_directories[strategy] = output_directory + strategy + '/'
        task_directories = os.listdir(directory)
        for task in task_directories:
            arff_file = os.path.join(result_directory, strategy, task, 'trace.arff')
            if os.path.isfile(arff_file):
                with open(arff_file, 'r') as fp:
                    trace_arff = arff.load(fp)
                trace = openml.runs.functions._create_trace_from_arff(trace_arff)
                all_traces[task][strategy] = trace
                openmlpimp.utils.obtain_performance_curves(trace, output_directory + strategy + '/' + task + '/')

                all_taskids.add(task)

    for task in all_traces.keys():
        openmlpimp.utils.boxplot_traces(all_traces[task], output_directory + 'boxplots/', str(task) + '.png')

    for task_id in all_taskids:
        openmlpimp.utils.plot_task(args.virtual_env, args.scripts_dir, strategy_directories, output_directory + 'plots/', task_id)

