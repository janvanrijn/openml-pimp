
import arff
import argparse
import math
import numpy as np
import openmlpimp
import os

def parse_args():
    all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest', 'sgd']
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--virtual_env', type=str, default=os.path.expanduser('~') + '/projects/pythonvirtual/plot2/bin/python', help='python virtual env for plotting')
    parser.add_argument('--scripts_dir', type=str, default=os.path.expanduser('~') + '/projects/plotting_scripts/scripts', help='directory to Katha\'s plotting scripts')
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/nemo/experiments/priorbased_experiments/', help='the directory to load the experiments from')

    return parser.parse_args()


def obtain_config(data_point, param_indices):
    # data_points = list<mixed>
    # param_indices = dict<int, str>
    config = []
    for key in sorted(param_indices):
        config.append(data_point[key])
    return tuple(config)


def check_iteration(data_points, param_indices, eval_idx):
    # data_points = list<list<mixed>>
    # param_indices = dict<int, str>

    num_steps = int(math.log(len(data_points), 2))

    # enumerates over trace backwards, checking whether
    # - configs in a step also appeared in the previous step
    # - these configs were indeed amongst the best half

    next_step_configs = {obtain_config(data_points[-1], param_indices)}
    for step in range(num_steps):

        current_configs = []
        current_scores = []
        for arms in range(2**(step+1), 2**(step+2)):
            current_configs.append(obtain_config(data_points[-arms], param_indices))
            current_scores.append(float(data_points[-arms][eval_idx]))

        possible_continue_arms = set()
        current_scores = np.array(current_scores, dtype=np.float)
        sorted = np.argsort(current_scores * -1)
        num_continue_arms = int(len(current_configs) / 2)
        for i in range(len(current_configs)):
            if i < num_continue_arms or current_scores[sorted[i]] == current_scores[sorted[num_continue_arms-1]]:
                possible_continue_arms.add(current_configs[sorted[i]])

        for config in next_step_configs:
            if config not in current_configs:
                raise ValueError('Could not find config %s' %str(config))

        if len(next_step_configs - possible_continue_arms) > 0:
            raise ValueError('Not correct arms continued. ')

        next_step_configs = set(current_configs)


def run():
    args = parse_args()

    for classifier in os.listdir(args.result_directory):
        if os.path.isfile(os.path.join(args.result_directory, classifier)):
            continue
        for fixed_parameters in os.listdir(os.path.join(args.result_directory, classifier)):
            print(openmlpimp.utils.get_time(), 'classifier:', classifier, fixed_parameters)
            directory = os.path.join(args.result_directory, classifier, fixed_parameters)
            clf_params = classifier + '__' + fixed_parameters
            for strategy in os.listdir(directory):
                for task_directory in os.listdir(os.path.join(directory, strategy)):
                    file = os.path.join(directory, strategy, task_directory, 'trace.arff')
                    task_id = int(task_directory)

                    if os.path.isfile(file):
                        arff_data = arff.load(open(file, 'r'))

                        param_indices = dict()
                        eval_index = None
                        for idx, attribute in enumerate(arff_data['attributes']):
                            if attribute[0].startswith('parameter_'):
                                param_indices[idx] = attribute[0]
                            elif attribute[0] == 'evaluation':
                                eval_index = idx

                        if len(param_indices) < 5:
                            raise ValueError()

                        # assumes order in the trace file..
                        current_repeat = 0
                        current_fold = 0
                        current_points = []
                        for datapoint in arff_data['data']:
                            repeat = int(datapoint[0])
                            fold = int(datapoint[1])
                            if repeat != current_repeat or fold != current_fold:
                                print('Checking %d %d with %d curves' % (repeat, fold, len(current_points)))
                                check_iteration(current_points, param_indices, eval_index)
                                current_repeat = repeat
                                current_fold = fold
                                current_points = []
                            current_points.append(datapoint)


if __name__ == '__main__':
    run()
