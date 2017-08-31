import argparse
import collections
import csv
import glob
import json
import openml
import os
import openmlpimp

plotting_virtual_env = '/home/vanrijn/projects/pythonvirtual/plot2/bin/python'
plotting_scripts_dir = '/home/vanrijn/projects/plotting_scripts/scripts'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--virtual_env', type=str, default=os.path.expanduser('~') + '/projects/pythonvirtual/plot2/bin/python', help='python virtual env for plotting')
    parser.add_argument('--scripts_dir', type=str, default=os.path.expanduser('~') + '/projects/plotting_scripts/scripts', help='directory to Katha\'s plotting scripts')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--flowid', type=int, default=7096, help="Flow id of optimizer") # 7089 beam_search(rf); 7096/7097 random_search(rf), 7116
    parser.add_argument('--classifier', type=str, default='random_forest', help="Classifier associated with the flow")
    parser.add_argument('--fixed_parameters', type=json.loads, default=None, help='Will only use configurations that have these parameters fixed')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/optimizers/randomsearch', help='the directory to store the results to')

    args = parser.parse_args()
    return args


def average_curves(file_pattern, save_directory, task_id, expected_files):
    curve_files = glob.glob(file_pattern)

    if len(curve_files) != expected_files:
        raise ValueError()

    avg_curve = collections.OrderedDict()

    for curve_file in curve_files:
        with open(curve_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                itt = int(row['iteration'])
                eval = float(row['evaluation'])
                if itt not in avg_curve:
                    avg_curve[itt] = 0.0
                avg_curve[itt] += eval / len(curve_files)

    try:
        os.makedirs(save_directory)
    except FileExistsError:
        pass

    with open(os.path.join(save_directory, str(task_id) + '.csv'), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['iteration', 'evaluation', 'evaluation2'])
        for idx in avg_curve.keys():
            csvwriter.writerow([idx + 1, avg_curve[idx], avg_curve[idx]])


def create_curve_files(output_directory, runids, classifier, exclude_param, num_values, all_task_ids):
    # creates the curve files for (one classifier,) all tasks, one exclude parameter, all possible values
    missing = {task_id: num_values for task_id in all_task_ids}

    for task_id in runids:
        all_values = openmlpimp.utils.get_param_values(classifier, exclude_param)
        unfinished = 0
        for value in all_values:
            task_directory = output_directory + '/' + exclude_param + '/' + str(task_id) + '/' + str(value)
            if os.path.isdir(task_directory):
                num_files = len(os.listdir(task_directory))
                if num_files != 10:
                    raise ValueError('Expected %d files, obtained %d, for task: %d' %(10, num_files, task_id))
            else:
                if value in runids[task_id]:
                    avg_task_directory = output_directory + '_avg/' + exclude_param + '/' + str(value)
                    openmlpimp.utils.obtain_performance_curves_openml(runids[task_id][value][0], task_directory, avg_task_directory, task_id)
                else:
                    unfinished += 1
        missing[task_id] = unfinished
    return missing


if __name__ == '__main__':
    args = parse_args()

    results_suffix = openmlpimp.utils.fixed_parameters_to_suffix(args.fixed_parameters)
    output_directory = args.output_directory + '/' + args.classifier + results_suffix

    openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server

    study = openml.study.get_study(args.openml_study)

    param_templates = dict()
    param_values = dict()
    for param in openmlpimp.utils.obtain_parameters(args.classifier):
        param_templates[param] = openmlpimp.utils.obtain_paramgrid(args.classifier, exclude=param)
        param_values[param] = openmlpimp.utils.get_param_values(args.classifier, param, args.fixed_parameters)

    results = openmlpimp.utils.obtain_runids(study.tasks, args.flowid, args.classifier, param_templates)

    all_taskids = set()
    strategy_curvesdir = dict()
    finished_tasks = set()
    missing_total = dict()
    for name, param_template in results.items():
        print(results[name])
        missing = create_curve_files(output_directory + '/curves', results[name], args.classifier, name, len(param_values[name]), study.tasks)
        for task_id in missing:
            if task_id not in missing_total:
                missing_total[task_id] = 0
            missing_total[task_id] += missing[task_id]

        all_taskids |= set(results[name].keys())
        strategy_curvesdir[name] = output_directory + '/curves/' + name

    for task_id in missing_total:
        print(task_id, missing_total[task_id])
        if missing_total[task_id] == 0:
            finished_tasks.add(task_id)
            dir = os.path.join(output_directory, 'curves_avg')
            for exclude_param in os.listdir(dir):
                pattern = dir + '/' + exclude_param + '/*/' + str(task_id) + '.csv'
                average_curves(pattern, output_directory + '/curves_avg_avg/' + exclude_param, task_id, len(param_values[exclude_param]))
            print('READY')
    print("total missing:", sum(missing_total.values()))

    # plot important ranks
    openmlpimp.utils.average_rank(args.virtual_env, args.scripts_dir, output_directory, output_directory + '/curves_avg_avg')

    for task_id in all_taskids:
        openmlpimp.utils.plot_task(plotting_virtual_env, plotting_scripts_dir, strategy_curvesdir, output_directory + '/plots/', task_id, wildcard_depth=2)

