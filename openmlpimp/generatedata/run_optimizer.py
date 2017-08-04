import argparse
import openml
import openmlpimp

from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from openmlpimp.sklearn.beam_search import BeamSearchCV


def parse_args():
    parser = argparse.ArgumentParser(description = 'Generate data for openml-pimp project')
    all_classifiers = ['adaboost', 'random_forest']
    parser.add_argument('--n_iters', type=int,  default=50, help='number of runs to be executed in case of random search. ')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')
    parser.add_argument('--array_index', type=int, required=True, help='the index of job array')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='random_forest', help='the classifier to execute')
    parser.add_argument('--reverse', type=bool, default=False, help='Threat the params in reverse order of importance')
    parser.add_argument('--optimizer', type=str, default='beam_search')

    args = parser.parse_args()
    return args


def obtain_paramgrid(classifier, reversed=False):
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

    if reversed:
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
    else:
        tasks = study.tasks[args.array_index]

    for task_id in tasks:
        try:
            task = openml.tasks.get_task(study.tasks[task_id])
            indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])

            if args.classifier == 'random_forest':
                pipeline = openmlpimp.utils.classifier_to_pipeline(RandomForestClassifier(random_state=1), indices)
            else:
                raise ValueError()

            if args.optimizer == 'beam_search':
                optimizer = BeamSearchCV(estimator=pipeline,
                                         param_distributions=obtain_paramgrid(args.classifier, args.reverse))
            else:
                raise ValueError()

            print(task.task_id)
            print(optimizer)
            run = openml.runs.run_model_on_task(task, optimizer)
            run.publish()
        except Exception as e:
            print(e)
