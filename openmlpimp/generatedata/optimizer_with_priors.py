import argparse
import json
import openml
import openmlpimp
import os
import random
import sklearn
import fasteners
import warnings

from sklearn.model_selection._search import RandomizedSearchCV
from openmlpimp.utils import SuccessiveHalving, HyperBand
from ConfigSpace.hyperparameters import NumericalHyperparameter


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                       'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                       'qda', 'random_forest', 'sgd']
    all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest', 'sgd']
    parser.add_argument('--cache_dir', type=str, default=os.path.expanduser('~') + '/experiments/cache_kde')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/priorbased_experiments')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks for the prior from')
    parser.add_argument('--flow_id', type=int, default=6970, help='the tag to obtain the tasks for the prior from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_taskid', type=int, nargs="+", default=None, help='the openml task id to execute')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='adaboost', help='the classifier to execute')
    parser.add_argument('--search_type', type=str, choices=['kde', 'uniform', 'empirical', 'multivariate'], default='kde', help='the way to apply the search')
    parser.add_argument('--bestN', type=int, default=10, help='number of best setups to consider for creating the priors')
    parser.add_argument('--fixed_parameters', type=json.loads, default=None, help='Will only use configurations that have these parameters fixed')
    parser.add_argument('--inverse_holdout', action="store_true", help='Will only operate on the task at hand (overestimate performance)')
    parser.add_argument('--ignore_logscale', action="store_true", help='Will only use hyperparameters that are not on a logscale')
    parser.add_argument('--oob_strategy', type=str, default='ignore', help='Way to handle priors that are out of bound')
    parser.add_argument('--n_executions', type=int, default=None, help='Max bound, for example for cluster jobs. ')
    parser.add_argument('--random_order', action="store_true", help='Iterates the tasks in a random order')
    parser.add_argument('--eta', type=int, default=2, help='successive halving parameter')
    parser.add_argument('--num_steps', type=int, default=5, help='successive halving parameter')
    parser.add_argument('--num_brackets', type=int, default=5, help='successive halving parameter')

    args = parser.parse_args()

    if args.search_type == 'uniform':
        if args.inverse_holdout is True:
            raise ValueError('Inverse holdout set not applicable to search type uniform')

    return args


def update_param_dist(classifier, param_distributions):
    param_dist_adjusted = dict()
    # TODO hacky update
    if classifier == 'adaboost':
        param_distributions['base_estimator__max_depth'] = param_distributions.pop('max_depth')

    for param_name, hyperparameter in param_distributions.items():
        if param_name == 'strategy':
            param_name = 'imputation__strategy'
        else:
            param_name = 'classifier__' + param_name
        param_dist_adjusted[param_name] = hyperparameter
    return param_dist_adjusted

if __name__ == '__main__':
    args = parse_args()
    if args.openml_server:
        openml.config.server = args.openml_server

    # select tasks to execute
    if args.openml_taskid is None:
        study = openml.study.get_study(args.study_id, 'tasks')
        all_task_ids = study.tasks
    elif isinstance(args.openml_taskid, int):
        all_task_ids = [args.openml_taskid]
    elif isinstance(args.openml_taskid, list):
        all_task_ids = args.openml_taskid
    else:
        raise ValueError()

    if args.random_order:
        random.shuffle(all_task_ids)

    optimizer_parameters = {}
    optimizer_parameters['bestN'] = args.bestN
    optimizer_parameters['inverse_holdout'] = args.inverse_holdout
    optimizer_parameters['ignore_logscale'] = args.ignore_logscale
    optimizer_parameters['oob_strategy'] = args.oob_strategy

    output_save_folder_suffix = openmlpimp.utils.fixed_parameters_to_suffix(optimizer_parameters)
    cache_save_folder_suffix = openmlpimp.utils.fixed_parameters_to_suffix(args.fixed_parameters)
    cache_dir = args.cache_dir + '/' + args.classifier + '/' + cache_save_folder_suffix

    configuration_space = openmlpimp.utils.get_config_space_casualnames(args.classifier, args.fixed_parameters)
    hyperparameters = dict(configuration_space._hyperparameters.items())
    if args.ignore_logscale:
        for param_name in hyperparameters.keys():
            if isinstance(hyperparameters[param_name], NumericalHyperparameter):
                hyperparameters[param_name].log = False

    print("classifier %s; flow id: %d; fixed_parameters: %s" %(args.classifier, args.flow_id, args.fixed_parameters))
    print("%s Tasks: %s" %(openmlpimp.utils.get_time(), str(all_task_ids)))
    executions_done = 0
    for task_id in all_task_ids:
        task = openml.tasks.get_task(task_id)
        data_name = task.get_dataset().name
        data_qualities = task.get_dataset().qualities
        print("%s Obtained task %d (%s); %s attributes; %s observations" % (openmlpimp.utils.get_time(), task_id,
                                                                            data_name,
                                                                            data_qualities['NumberOfFeatures'],
                                                                            data_qualities['NumberOfInstances']))

        indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
        base, required_params = openmlpimp.utils.modeltype_to_classifier(args.classifier, {'random_state': 1})
        pipe = openmlpimp.utils.classifier_to_pipeline(base, indices)
        if required_params is not None:
            pipe.set_params(**required_params)

        output_dir = args.output_dir + '/' + args.classifier + cache_save_folder_suffix + '/' + args.search_type + '__' + output_save_folder_suffix[1:] + '/' + str(task_id)
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass

        expected_path = output_dir + '/trace.arff'
        if os.path.isfile(expected_path):
            print("Task already finished: %d %s (%s)" % (task_id, args.search_type, expected_path))
            continue

        lock_file = fasteners.InterProcessLock(output_dir + '/tmp.lock')
        obtained_lock = lock_file.acquire(blocking=False)
        try:
            if not obtained_lock:
                # this means some other process already is working
                print("Task already in progress: %d %s" %(task_id, args.search_type))
                continue

            if args.inverse_holdout:
                holdout = set(all_task_ids) - {task_id}
            else:
                holdout = {task_id}

            if args.search_type == 'kde':  # KDE
                param_distributions = openmlpimp.utils.get_kde_paramgrid(cache_dir,
                                                                         args.study_id,
                                                                         args.flow_id,
                                                                         hyperparameters,
                                                                         args.fixed_parameters,
                                                                         holdout=holdout,
                                                                         bestN=args.bestN,
                                                                         oob_strategy=args.oob_strategy)
                param_distributions = update_param_dist(args.classifier, param_distributions)
                print('%s Param Grid:' % openmlpimp.utils.get_time(), param_distributions)

            elif args.search_type == 'empirical':
                param_distributions = openmlpimp.utils.get_empericaldistribution_paramgrid(cache_dir,
                                                                                           args.study_id,
                                                                                           args.flow_id,
                                                                                           hyperparameters,
                                                                                           args.fixed_parameters,
                                                                                           holdout=holdout,
                                                                                           bestN=args.bestN)
                param_distributions = update_param_dist(args.classifier, param_distributions)
                print('%s Param Grid:' % openmlpimp.utils.get_time(), param_distributions)

            elif args.search_type == 'uniform':
                param_distributions = openmlpimp.utils.get_uniform_paramgrid(hyperparameters, args.fixed_parameters)
                param_distributions = update_param_dist(args.classifier, param_distributions)
                print('%s Param Grid:' % openmlpimp.utils.get_time(), param_distributions)

            else:
                raise ValueError()

            print('%s Start modelling ... [takes a while]' %openmlpimp.utils.get_time())

            # TODO: make this better
            fixed_param_values = dict()
            if args.fixed_parameters is not None:
                for param_name, value in args.fixed_parameters.items():
                    param_name = 'estimator__classifier__' + param_name
                    fixed_param_values[param_name] = value

            if args.num_brackets is not None:
                optimizer = HyperBand(estimator=pipe,
                                      param_distributions=param_distributions,
                                      random_state=1,
                                      n_jobs=-1,
                                      eta=args.eta,
                                      num_brackets=args.num_brackets)
            elif args.num_steps is not None:
                optimizer = SuccessiveHalving(estimator=pipe,
                                              param_distributions=param_distributions,
                                              random_state=1,
                                              n_jobs=-1,
                                              eta=args.eta,
                                              num_steps=args.num_steps)
            optimizer.set_params(**fixed_param_values)
            print("%s Optimizer: %s" %(openmlpimp.utils.get_time(), str(optimizer)))
            print("%s Steps: " %openmlpimp.utils.get_time(), optimizer.estimator.steps)

            run = openmlpimp.utils.do_run(task, optimizer, output_dir, False)
            score = run.get_metric_fn(sklearn.metrics.accuracy_score)

            print('%s [SCORE] Data: %s; Accuracy: %0.2f' % (openmlpimp.utils.get_time(), task.get_dataset().name, score.mean()))

            executions_done += 1
            if args.n_executions is not None:
                if executions_done >= args.n_executions:
                    break
        finally:
            if obtained_lock:
                lock_file.release()
    print("%s Executions done: %d" % (openmlpimp.utils.get_time(), executions_done))
