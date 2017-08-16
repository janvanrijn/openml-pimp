import argparse
import openml
import collections
import subprocess
import json
import csv
import os

from collections import defaultdict

from openmlpimp.generatedata.run_optimizer import obtain_paramgrid


plotting_virtual_env = '/home/vanrijn/projects/pythonvirtual/plot2/bin/python'
plotting_scripts_dir = '/home/vanrijn/projects/plotting_scripts/scripts'


def parse_args():
    parser = argparse.ArgumentParser(description = 'Generate data for openml-pimp project')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--flowid', type=int, required=True, help="Flow id of optimizer") # 7089 rf

    args = parser.parse_args()
    return args


def obtain_runids(task_ids, classifier):
    param_order_normal  = obtain_paramgrid(classifier, False)
    param_order_reverse = obtain_paramgrid(classifier, True)

    decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)

    runids_normal = defaultdict(list)
    runids_reverse = defaultdict(list)
    runids_other = defaultdict(list)

    for task_id in task_ids:
        print("task", task_id)
        try:
            runs = openml.runs.list_runs(task=[task_id], flow=[args.flowid])
        except:
            print("runs None")
            continue

        for run_id in runs:
            setup_id = runs[run_id]['setup_id']
            if setup_id not in setups:
                raise ValueError()
            for idx, parameter in setups[setup_id].parameters.items():
                if parameter.parameter_name == 'param_distributions':
                    order = decoder.decode(parameter.value)
                    if dict(order) != dict(param_order_normal):
                        print('run %d: wrong parameter values' %run_id)
                        continue

                    if order == param_order_normal:
                        runids_normal[task_id].append(run_id)
                    elif order == param_order_reverse:
                        runids_reverse[task_id].append(run_id)
                    else:
                        runids_other[task_id].append(run_id)
    return runids_normal, runids_reverse, runids_other


def plot_task(strategy_directories, plot_directory, task_id):
    # make regret plot:
    script = "%s %s/plot_test_performance_from_csv.py " % (plotting_virtual_env, plotting_scripts_dir)
    cmd = [script]
    for strategy in strategy_directories:
        strategy_splitted = strategy.split('/')
        cmd.append(strategy_splitted[-2])
        cmd.append(strategy + str(task_id) + '/*.csv')

    try:
        os.makedirs(plot_directory)
    except FileExistsError:
        pass

    cmd.append('--save %s ' % os.path.join(plot_directory, 'validation_regret%d.png' %task_id))
    cmd.append('--ylabel "Accuracy Loss"')

    subprocess.run(' '.join(cmd), shell=True)
    print('CMD: ', ' '.join(cmd))


def obtain_performance_curves(run_id, directory, improvements=True):
    trace = openml.runs.get_run_trace(run_id)
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


def create_curve_files(runids, dirname):

    for task_id in runids:
        task_directory = output_directory + dirname + '/' + str(task_id) + '/'
        if os.path.isdir(task_directory):
            num_files = len(os.listdir(task_directory))
            if num_files != 10:
                raise ValueError('Expected %d files, obtained %d, for task: %d' %(10, num_files, task_id))
        else:
            obtain_performance_curves(runids[task_id][0], task_directory)


if __name__ == '__main__':
    output_directory = '/home/vanrijn/experiments/optimizers/'
    args = parse_args()
    openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server

    study = openml.study.get_study(args.openml_study)
    setups = openml.setups.list_setups(flow=args.flowid)
    classifier = 'random_forest'  # TODO

    runids_normal, runids_reverse, runids_other = obtain_runids(study.tasks, classifier)

    print(runids_normal)
    print(runids_reverse)

    create_curve_files(runids_normal, 'normal')
    create_curve_files(runids_reverse, 'reverse')

    for task_id in runids_normal:
        plot_task([output_directory + 'normal/', output_directory + 'reverse/'], output_directory + 'plots/', task_id)