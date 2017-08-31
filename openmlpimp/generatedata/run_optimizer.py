import argparse
import json
import openml
import openmlpimp
import random

from sklearn.model_selection._search import RandomizedSearchCV


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['adaboost', 'random_forest', 'libsvm_svc']
    parser.add_argument('--n_iters', type=int, default=50, help='number of runs to be executed in case of random search')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')
    parser.add_argument('--array_index', type=int, help='the index of job array')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='adaboost', help='the classifier to execute')
    parser.add_argument('--fixed_parameters', type=json.loads, default=None, help='Will only use configurations that have these parameters fixed')

    parser.add_argument('--optimizer', type=str, default='random_search')

    args = parser.parse_args()
    return args


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

        preset_params = {'random_state': 1}
        if args.fixed_parameters:
            preset_params.update(args.fixed_parameters)
        if args.classifier == 'adaboost':
            preset_params['base_estimator__random_state'] = 1
        classifier, required_params = openmlpimp.utils.modeltype_to_classifier(args.classifier, preset_params)
        pipeline = openmlpimp.utils.classifier_to_pipeline(classifier, indices)
        pipeline.set_params(**required_params)

        print("task", task.task_id)
        if args.optimizer == 'random_search':
            all_exclusion_sets = list(openmlpimp.utils.obtain_parameters(args.classifier))

            random.shuffle(all_exclusion_sets)
            for exclude_param in all_exclusion_sets:
                param_distributions = openmlpimp.utils.obtain_paramgrid(args.classifier, exclude=exclude_param)
                print("param grid", param_distributions.keys())
                values = openmlpimp.utils.get_param_values(args.classifier, exclude_param, args.fixed_parameters)
                random.shuffle(values)
                for value in values:
                    print("exclude", exclude_param, value)
                    optimizer = RandomizedSearchCV(estimator=pipeline,
                                                   param_distributions=param_distributions,
                                                   n_iter=args.n_iters,
                                                   random_state=1,
                                                   n_jobs=-1)
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
