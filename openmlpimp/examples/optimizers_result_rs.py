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
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/nemo/experiments/rs_experiments/', help='the directory to load the experiments from')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/optimizers/random_search/', help='the directory to store the results to')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')

    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='libsvm_svc', help='the classifier to execute')
    parser.add_argument('--fixed_parameters', type=json.loads, default={'kernel': 'sigmoid'}, help='Will only use configurations that have these parameters fixed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    study = openml.study.get_study(args.openml_study)

    results_suffix = openmlpimp.utils.fixed_parameters_to_suffix(args.fixed_parameters)

    result_directory = args.result_directory + args.classifier + '/' + results_suffix
    output_directory = args.output_directory + args.classifier + '/' + results_suffix

    all_exclude_params = os.listdir(result_directory)
    strategy_directories = {}

    for task in study.tasks:
        for exclude_param in all_exclude_params:
            strategy_directories[exclude_param] = os.path.join(output_directory + '/curves', exclude_param)
            expected_values = openmlpimp.utils.get_param_values(args.classifier, exclude_param, args.fixed_parameters)
            all_exclude_values = os.listdir(os.path.join(result_directory, exclude_param))
            trace_count = 0
            traces = []
            for value in all_exclude_values:
                arff_file = os.path.join(result_directory, exclude_param, value, str(task), 'trace.arff')
                if os.path.isfile(arff_file):
                    trace_count += 1
                    with open(arff_file, 'r') as fp:
                        trace_arff = arff.load(fp)
                    traces.append(openml.runs.functions._create_trace_from_arff(trace_arff))

            if len(traces) == len(expected_values):
                output_indivudual = output_directory + '/curves/' + exclude_param + '/' + str(task)
                output_averaged = output_directory + '/curves_avg/' + exclude_param
                openmlpimp.utils.obtain_performance_curves(traces, output_indivudual, output_averaged, task)

    # plot all ranks
    openmlpimp.utils.average_rank(args.virtual_env, args.scripts_dir, output_directory, output_directory + '/curves_avg')

    for task_id in study.tasks:
        openmlpimp.utils.plot_task(args.virtual_env, args.scripts_dir, strategy_directories, output_directory + '/plots', task_id)

