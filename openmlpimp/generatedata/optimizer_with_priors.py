import argparse
import arff
import copy
import json
import openml
import openmlpimp
import os
import sklearn
import fasteners

import autosklearn.constants
from autosklearn.util.pipeline import get_configuration_space
from sklearn.model_selection._search import RandomizedSearchCV


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                       'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                       'qda', 'random_forest', 'sgd']
    all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest', 'sgd']
    parser.add_argument('--cache_dir', type=str, default=os.path.expanduser('~') + '/experiments/cache_kde')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/random_search_prior')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks for the prior from')
    parser.add_argument('--flow_id', type=int, default=6970, help='the tag to obtain the tasks for the prior from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_taskid', type=int, nargs="+", default=None, help='the openml task id to execute')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='adaboost', help='the classifier to execute')
    parser.add_argument('--search_type', type=str, choices=['priors', 'uniform'], default=None, help='the way to apply the search')
    parser.add_argument('--n_iters', type=int, default=50, help='number of runs to be executed in case of random search')
    parser.add_argument('--bestN', type=int, default=10, help='number of best setups to consider for creating the priors')
    parser.add_argument('--fixed_parameters', type=json.loads, default=None, help='Will only use configurations that have these parameters fixed')

    args = parser.parse_args()
    return args


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

    # select search_types to execute
    search_types = [args.search_type]
    if args.search_type is None:
        search_types = ['priors', 'uniform']

    important_parameters = copy.deepcopy(args.fixed_parameters) if args.fixed_parameters is not None else {}
    important_parameters['bestN'] = args.bestN

    save_folder_suffix = openmlpimp.utils.fixed_parameters_to_suffix(important_parameters)
    cache_dir = args.cache_dir + '/' + args.classifier + '/' + save_folder_suffix

    configuration_space = get_configuration_space(
        info={'task': autosklearn.constants.MULTICLASS_CLASSIFICATION, 'is_sparse': 0},
        include_estimators=[args.classifier],
        include_preprocessors=['no_preprocessing'])

    print("classifier %s; flow id: %d; fixed_parameters: %s" %(args.classifier, args.flow_id, args.fixed_parameters))
    print("%s Tasks: %s" %(openmlpimp.utils.get_time(), str(all_task_ids)))
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

        hyperparameters = openmlpimp.utils.configspace_to_relevantparams(configuration_space)

        for search_type in search_types:
            output_dir = args.output_dir + '/' + args.classifier + '/' + save_folder_suffix + '/' + search_type + '/' + str(task_id) + '/'
            try:
                os.makedirs(output_dir)
            except FileExistsError:
                pass

            if os.path.isfile(output_dir + 'trace.arff'):
                print("Task already finished: %d %s" % (task_id, search_type))
                continue

            lock_file = fasteners.InterProcessLock(output_dir + 'tmp.lock')
            obtained_lock = lock_file.acquire(blocking=False)
            try:
                if not obtained_lock:
                    # this means some other process already is working
                    print("Task already in progress: %d %s" %(task_id, search_type))
                    continue

                if search_type == 'priors':
                    param_distributions = openmlpimp.utils.get_prior_paramgrid(cache_dir,
                                                                               args.study_id,
                                                                               args.flow_id,
                                                                               hyperparameters,
                                                                               args.fixed_parameters,
                                                                               holdout=task_id,
                                                                               bestN=args.bestN)
                    # TODO: hacky mapping update
                    if args.classifier == 'adaboost':
                        param_distributions['base_estimator__max_depth'] = param_distributions.pop('max_depth')


                elif search_type == 'uniform':
                    param_distributions = openmlpimp.utils.get_uniform_paramgrid(hyperparameters, args.fixed_parameters)
                else:
                    raise ValueError()
                print('%s Param Grid:' %openmlpimp.utils.get_time(), param_distributions)
                print('%s Start modelling ... [takes a while]' %openmlpimp.utils.get_time())

                # TODO: make this better
                param_dist_adjusted = dict()
                fixed_param_values = dict()
                for param_name, hyperparameter in param_distributions.items():
                    if param_name == 'strategy':
                        param_name = 'imputation__strategy'
                    else:
                        param_name = 'classifier__' + param_name
                    param_dist_adjusted[param_name] = hyperparameter
                if args.fixed_parameters is not None:
                    for param_name, value in args.fixed_parameters.items():
                        param_name = 'estimator__classifier__' + param_name
                        fixed_param_values[param_name] = value

                optimizer = RandomizedSearchCV(estimator=pipe,
                                               param_distributions=param_dist_adjusted,
                                               n_iter=args.n_iters,
                                               random_state=1,
                                               n_jobs=-1)
                optimizer.set_params(**fixed_param_values)
                print("%s Optimizer: %s" %(openmlpimp.utils.get_time(), str(optimizer)))

                res = openml.runs.functions._run_task_get_arffcontent(optimizer, task, task.class_labels)
                run = openml.runs.OpenMLRun(task_id=task.task_id, dataset_id=None, flow_id=None,
                                            model=optimizer)
                run.data_content, run.trace_content, run.trace_attributes, run.fold_evaluations, _ = res
                score = run.get_metric_fn(sklearn.metrics.accuracy_score)

                print('%s [SCORE] Data: %s; Accuracy: %0.2f' % (openmlpimp.utils.get_time(), task.get_dataset().name, score.mean()))

                trace_arff = arff.dumps(run._generate_trace_arff_dict())
                with open(output_dir + 'trace.arff', 'w') as f:
                    f.write(trace_arff)
            finally:
                if obtained_lock:
                    lock_file.release()
