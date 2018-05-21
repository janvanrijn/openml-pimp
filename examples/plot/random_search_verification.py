import argparse
import arff
import collections
import copy
import csv
import json
import io
import os
import openml
import openmlpimp
import pandas as pd
import pickle


# Mounting CMD: sshfs fr_jv1031@login1.nemo.uni-freiburg.de:/home/fr/fr_fr/fr_jv1031 ~/nemo/
def parse_args():
    all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest', 'sgd']
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--virtual_env', type=str, default=os.path.expanduser('~') + '/anaconda3/envs/openmlpimp/bin/python', help='python virtual env for plotting')
    parser.add_argument('--scripts_dir', type=str, default=os.path.expanduser('~') + '/projects/plotting_scripts/scripts', help='directory to Katha\'s plotting scripts')
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/uni_freiburg_experiments/openml-pimp/', help='the directory to load the experiments from')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/pimp/optimizers/random_search/', help='the directory to store the results to')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')

    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='libsvm_svc', help='the classifier to execute')
    parser.add_argument('--fixed_parameters', type=json.loads, default={'kernel': 'rbf'}, help='Will only use configurations that have these parameters fixed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    study = openml.study.get_study(args.openml_study, 'tasks')

    results_suffix = openmlpimp.utils.fixed_parameters_to_suffix(args.fixed_parameters)

    result_directory = args.result_directory + args.classifier + '/' + results_suffix
    output_directory = args.output_directory + args.classifier + '/' + results_suffix

    all_exclude_params = os.listdir(result_directory)
    strategy_directories = {}

    missing = 0
    flow_id = None
    task_missing = collections.defaultdict(int)

    all_columns = copy.deepcopy(all_exclude_params)
    all_columns.append('task_id')
    dataframe = pd.DataFrame(columns=all_columns)

    for task in study.tasks:
        current_row = {'task_id': task}
        for exclude_param in all_exclude_params:
            name = openmlpimp.utils.name_mapping(args.classifier, exclude_param, replace_underscores=False)
            strategy_directories[name] = os.path.join(output_directory + '/curves', name)
            expected_values = openmlpimp.utils.get_param_values(args.classifier, exclude_param, args.fixed_parameters)
            all_exclude_values = os.listdir(os.path.join(result_directory, exclude_param))
            trace_count = 0
            traces = []
            for value in all_exclude_values:
                xml_file = os.path.join(result_directory, exclude_param, value, str(task), 'run.xml')
                arff_file = os.path.join(result_directory, exclude_param, value, str(task), 'trace.arff')
                if os.path.isfile(arff_file) and os.path.isfile(xml_file):
                    trace_count += 1
                    with open(xml_file, 'r') as fp:
                        with io.open(xml_file, encoding='utf8') as fh:
                            run = openml.runs.functions._create_run_from_xml(xml=fh.read(), from_server=False)
                        if flow_id is None:
                            flow_id = run.flow_id
                        if flow_id != run.flow_id:
                            raise ValueError('Flow ids do not match: %d vs %d' %(flow_id, run.flow_id))
                    with open(arff_file, 'r') as fp:
                        trace_arff = arff.load(fp)
                    traces.append(openml.runs.functions._create_trace_from_arff(trace_arff))
                else:
                    missing += 1
                    task_missing[task] += 1

            if len(traces) == len(expected_values):
                output_indivudual = output_directory + '/curves/' + name + '/' + str(task)
                output_averaged = output_directory + '/curves_avg/' + name
                openmlpimp.utils.obtain_performance_curves(traces, output_indivudual, output_averaged, task, inverse=True)

                average_curve = output_averaged + '/' + str(task) + '.csv'
                with open(average_curve) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        pass
                    # row now contains the last (final) value
                    current_row[exclude_param] = row['evaluation']
        dataframe = dataframe.append(current_row, ignore_index=True)
    dataframe.set_index('task_id')

    print(args.classifier, args.fixed_parameters)
    print('Flow id: %d' %flow_id)
    print(task_missing)
    print('Incomplete tasks:', len(task_missing))
    print('Total missing', missing)

    pickle.dump(dataframe, open(output_directory + '/random_search.pkl', 'wb'))

    # plot all ranks
    openmlpimp.utils.average_rank(args.virtual_env, args.scripts_dir, output_directory, output_directory + '/curves_avg', ylabel="Average Rank")

    # for task_id in study.tasks:
    #     openmlpimp.utils.plot_task(args.virtual_env, args.scripts_dir, strategy_directories, output_directory + '/plots', task_id)
    #
