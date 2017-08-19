import argparse
import openml
import openmlpimp
import random

from collections import OrderedDict

from sklearn.model_selection._search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from openmlpimp.sklearn.beam_search import BeamSearchCV


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['adaboost', 'random_forest']
    parser.add_argument('--n_iters', type=int, default=50, help='number of runs to be executed in case of random search')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')
    parser.add_argument('--array_index', type=int, help='the index of job array')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='random_forest', help='the classifier to execute')
    parser.add_argument('--optimizer', type=str, default='random_search')
    parser.add_argument('--num_exclusions', type=int, default=1, help='Only applicable for Random Search')

    args = parser.parse_args()
    return args


def obtain_parameters(classifier):
    return set(obtain_paramgrid(classifier).keys())


def obtain_parameter_combinations(classifier, num_params):
    if num_params != 2:
        raise ValueError('Not implemented yet')
    result = list()
    params = set(obtain_paramgrid(classifier).keys())
    for param1 in params:
        for param2 in params:
            if param1 == param2:
                continue
            result.append([param1, param2])
    return result


def get_param_values(classifier, parameter):
    param_grid = obtain_paramgrid(classifier)
    return param_grid[parameter]


def obtain_paramgrid(classifier, exclude=None, reverse=False):
    if classifier == 'random_forest':
        param_grid = OrderedDict()
        param_grid['classifier__min_samples_leaf'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        param_grid['classifier__max_features'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        param_grid['classifier__bootstrap'] = [True, False]
        param_grid['classifier__min_samples_split'] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        param_grid['classifier__criterion'] = ['gini', 'entropy']
        param_grid['imputation__strategy'] = ['mean','median','most_frequent']
    else:
        raise ValueError()

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        for exclude_param in exclude:
            if exclude_param not in param_grid.keys():
                raise ValueError()
            del param_grid[exclude_param]

    if reverse:
        return OrderedDict(reversed(list(param_grid.items())))
    else:
        return param_grid


if __name__ == '__main__':
    args = parse_args()
    openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server

    study = openml.study.get_study(args.openml_study)
    if args.array_index is None:
        tasks = study.tasks
        random.shuffle(tasks)
    else:
        tasks = [study.tasks[args.array_index]]

    for task_id in tasks:
        task = openml.tasks.get_task(task_id)
        indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])

        if args.classifier == 'random_forest':
            pipeline = openmlpimp.utils.classifier_to_pipeline(RandomForestClassifier(random_state=1), indices)
        else:
            raise ValueError()

        print("task", task.task_id)
        if args.optimizer == 'beam_search':
            param_distributions = obtain_paramgrid(args.classifier)
            optimizer = BeamSearchCV(estimator=pipeline,
                                     param_distributions=param_distributions)
            print(optimizer)
            print(param_distributions.keys())
            try:
                run = openml.runs.run_model_on_task(task, optimizer)
                run.publish()
            except Exception as e:
                print(e)
        elif args.optimizer == 'random_search':
            if args.num_exclusions == 1:
                all_exclusion_sets = list(obtain_parameters(args.classifier))
            else:
                all_exclusion_sets = obtain_parameter_combinations(args.classifier, args.num_exclusions)

            random.shuffle(all_exclusion_sets)
            for exclude_param in all_exclusion_sets:
                param_distributions = obtain_paramgrid(args.classifier, exclude=exclude_param)
                print("param grid", param_distributions.keys())
                values = get_param_values(args.classifier, exclude_param)
                random.shuffle(values)
                for value in values:
                    print("exclude", exclude_param, value)
                    optimizer = RandomizedSearchCV(estimator=pipeline,
                                                   param_distributions=param_distributions,
                                                   n_iter=args.n_iters,
                                                   random_state=1)
                    optimizer.set_params(**{'estimator__' + exclude_param: value})
                    print(optimizer)
                    try:
                        run = openml.runs.run_model_on_task(task, optimizer)
                        run.publish()
                        print('uploaded with id %d' %run.run_id)
                    except Exception as e:
                        print(e)
        else:
            raise ValueError('unknown optimizer.')
