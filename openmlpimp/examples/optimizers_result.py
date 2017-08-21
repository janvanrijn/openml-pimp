import argparse
import openml
import collections
import subprocess
import csv
import os
import sys
import openmlpimp

from collections import defaultdict


plotting_virtual_env = '/home/vanrijn/projects/pythonvirtual/plot2/bin/python'
plotting_scripts_dir = '/home/vanrijn/projects/plotting_scripts/scripts'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--flowid', type=int, required=True, help="Flow id of optimizer") # 7089 beam_search(rf); 7096/7097 random_search(rf)

    args = parser.parse_args()
    return args


def plot_task(strategy_directories, plot_directory, task_id):
    # make regret plot:
    script = "%s %s/plot_test_performance_from_csv.py " % (plotting_virtual_env, plotting_scripts_dir)
    cmd = [script]
    for strategy in strategy_directories:
        strategy_splitted = strategy.split('/')
        cmd.append(strategy_splitted[-2])
        cmd.append(strategy + str(task_id) + '/*/*.csv')
    try:
        os.makedirs(plot_directory)
    except FileExistsError:
        pass

    cmd.append('--save %s ' % os.path.join(plot_directory, 'validation_regret%d.png' %task_id))
    cmd.append('--ylabel "Accuracy Loss"')

    subprocess.run(' '.join(cmd), shell=True)
    print('CMD: ', ' '.join(cmd))


def obtain_performance_curves(run_id, directory, improvements=True):
    try:
        trace = openml.runs.get_run_trace(run_id)
    except Exception as e:
        sys.stderr.write(e.message)
        return

    curves = defaultdict(dict)

    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    for itt in trace.trace_iterations:
        cur = trace.trace_iterations[itt]
        curves[(cur.repeat, cur.fold)][cur.iteration] = cur.evaluation

    for curve in curves.keys():
        current_curve = curves[curve]
        curves[curve] = list(collections.OrderedDict(sorted(current_curve.items())).values())

    if improvements:
        for curve in curves.keys():
            current_curve = curves[curve]
            for idx in range(1, len(current_curve)):
                if current_curve[idx] < current_curve[idx-1]:
                    current_curve[idx] = current_curve[idx - 1]

    for repeat, fold in curves.keys():
        with open(directory + '%d_%d.csv' %(repeat, fold), 'w') as csvfile:
            current_curve = curves[(repeat, fold)]
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['iteration', 'evaluation', 'evaluation2'])
            for idx in range(len(current_curve)):
                csvwriter.writerow([idx+1, current_curve[idx], current_curve[idx]])


def create_curve_files(runids, classifier, exclude_param):
    missing = dict()
    for task_id in runids:
        all_values = openmlpimp.utils.get_param_values(classifier, exclude_param)
        unfinished = 0
        for value in all_values:
            task_directory = output_directory + exclude_param + '/' + str(task_id) + '/' + str(value) + '/'
            if os.path.isdir(task_directory):
                num_files = len(os.listdir(task_directory))
                if num_files != 10:
                    raise ValueError('Expected %d files, obtained %d, for task: %d' %(10, num_files, task_id))
            else:
                if value in runids[task_id]:
                    obtain_performance_curves(runids[task_id][value][0], task_directory)
                else:
                    unfinished += 1
        missing[task_id] = unfinished
    return missing


if __name__ == '__main__':
    output_directory = '/home/vanrijn/experiments/optimizers/randomsearch/'
    args = parse_args()
    openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server

    study = openml.study.get_study(args.openml_study)
    classifier = 'random_forest'  # TODO

    # param_templates = {'normal': obtain_paramgrid(classifier, reverse=False),
    #                    'reverse': obtain_paramgrid(classifier, reverse=True)}
    param_templates = dict()
    for param in openmlpimp.utils.obtain_parameters(classifier):
        param_templates[param] = openmlpimp.utils.obtain_paramgrid(classifier, exclude=param)

    results = openmlpimp.utils.obtain_runids(study.tasks, args.flowid, classifier, param_templates)

    all_taskids = set()
    all_strategies = list()
    missing_total = dict()
    for name, param_template in results.items():
        print(results[name])
        missing = create_curve_files(results[name], classifier, name)
        for task_id in missing:
            if task_id not in missing_total:
                missing_total[task_id] = 0
            missing_total[task_id] += missing[task_id]

        all_taskids |= set(results[name].keys())
        all_strategies.append(output_directory + name + '/')

    for task_id in missing_total:
        print(task_id, missing_total[task_id])
    print("total missing:", sum(missing_total.values()))

    for task_id in all_taskids:
        plot_task(all_strategies, output_directory + 'plots/', task_id)

