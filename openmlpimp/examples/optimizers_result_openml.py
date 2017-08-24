import argparse
import openml
import os
import openmlpimp

plotting_virtual_env = '/home/vanrijn/projects/pythonvirtual/plot2/bin/python'
plotting_scripts_dir = '/home/vanrijn/projects/plotting_scripts/scripts'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--flowid', type=int, required=True, help="Flow id of optimizer") # 7089 beam_search(rf); 7096/7097 random_search(rf)

    args = parser.parse_args()
    return args


def create_curve_files(output_directory, runids, classifier, exclude_param):
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
                    openmlpimp.utils.obtain_performance_curves_openml(runids[task_id][value][0], task_directory)
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
    all_strategies = dict()
    missing_total = dict()
    for name, param_template in results.items():
        print(results[name])
        missing = create_curve_files(output_directory, results[name], classifier, name)
        for task_id in missing:
            if task_id not in missing_total:
                missing_total[task_id] = 0
            missing_total[task_id] += missing[task_id]

        all_taskids |= set(results[name].keys())
        all_strategies[name] = output_directory + name + '/'

    for task_id in missing_total:
        print(task_id, missing_total[task_id])
    print("total missing:", sum(missing_total.values()))

    for task_id in all_taskids:
        openmlpimp.utils.plot_task(plotting_virtual_env, plotting_scripts_dir, all_strategies, output_directory + 'plots/', task_id)

