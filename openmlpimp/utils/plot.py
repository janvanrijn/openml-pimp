import arff
import collections
import csv
import openml
import os
import subprocess
import sys


def to_csv_file(ranks_dict, location):
    hyperparameters = None
    for task_id, params in ranks_dict.items():
        hyperparameters = set(params)

    with open(location, 'w') as csvfile:
        fieldnames = ['task_id']
        fieldnames.extend(hyperparameters)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task_id, param_values in ranks_dict.items():
            result = {}
            result.update(param_values)
            result['task_id'] = 'Task %d' %task_id
            writer.writerow(result)
    pass


def to_csv_unpivot(ranks_dict, location):
    with open(location, 'w') as csvfile:
        fieldnames = ['task_id', 'param_id', 'param_name', 'variance_contribution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for task_id, param_values in ranks_dict.items():

            for param_name, variance_contribution in param_values.items():
                result = {'task_id' : task_id,
                          'param_id': param_name,
                          'param_name': param_name,
                          'variance_contribution': variance_contribution}
                writer.writerow(result)
    pass


def plot_task(plotting_virtual_env, plotting_scripts_dir, strategy_directory, plot_directory, task_id):
    # make regret plot:
    script = "%s %s/plot_test_performance_from_csv.py " % (plotting_virtual_env, plotting_scripts_dir)
    cmd = [script]
    for strategy, directory in strategy_directory.items():
        cmd.append(strategy)
        cmd.append(directory + str(task_id) + '/*.csv')
    try:
        os.makedirs(plot_directory)
    except FileExistsError:
        pass

    cmd.append('--save %s ' % os.path.join(plot_directory, 'validation_regret%s.png' %str(task_id)))
    cmd.append('--ylabel "Accuracy Loss"')

    subprocess.run(' '.join(cmd), shell=True)
    print('CMD: ', ' '.join(cmd))


def obtain_performance_curves(trace, save_directory, improvements):
    curves = collections.defaultdict(dict)

    try:
        os.makedirs(save_directory)
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
        with open(save_directory + '%d_%d.csv' %(repeat, fold), 'w') as csvfile:
            current_curve = curves[(repeat, fold)]
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['iteration', 'evaluation', 'evaluation2'])
            for idx in range(len(current_curve)):
                csvwriter.writerow([idx+1, current_curve[idx], current_curve[idx]])


def obtain_performance_curves_arff(arff_location, save_directory, improvements=True):
    with open(arff_location, 'r') as arff_file:
        trace_arff = arff.load(arff_file)
    trace = openml.runs.functions._create_trace_from_arff(trace_arff)
    obtain_performance_curves(trace, save_directory, improvements)


def obtain_performance_curves_openml(run_id, save_directory, improvements=True):
    try:
        trace = openml.runs.get_run_trace(run_id)
    except Exception as e:
        sys.stderr.write(e.message)
        return
    obtain_performance_curves(trace, save_directory, improvements)

