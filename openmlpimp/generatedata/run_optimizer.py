import argparse
import arff
import fasteners
import json
import openml
import openmlpimp
import os
import random
import sklearn

from sklearn.model_selection._search import RandomizedSearchCV


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['adaboost', 'random_forest', 'libsvm_svc']
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/rs_experiments')
    parser.add_argument('--n_iters', type=int, default=50, help='number of runs to be executed in case of random search')
    parser.add_argument('--openml_study', type=str, default='OpenML100', help='the study to obtain the tasks from')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='random_forest', help='the classifier to execute')
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
    cache_save_folder_suffix = openmlpimp.utils.fixed_parameters_to_suffix(args.fixed_parameters)

    for task_id in study.tasks:
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
            all_exclusion_sets = list(openmlpimp.utils.obtain_parameters(args.classifier, args.fixed_parameters))

            # random.shuffle(all_exclusion_sets)
            for exclude_param in all_exclusion_sets:
                param_distributions = openmlpimp.utils.obtain_paramgrid(args.classifier, exclude=exclude_param, fixed_parameters=args.fixed_parameters)
                print("param grid", param_distributions.keys())
                values = openmlpimp.utils.get_param_values(args.classifier, exclude_param, args.fixed_parameters)
                # random.shuffle(values)
                for value in values:
                    print("exclude", exclude_param, value, type(value))
                    output_dir = args.output_dir + '/' + args.classifier + cache_save_folder_suffix + '/' + exclude_param + '/' + str(value) + '/' + str(task_id)
                    try:
                        os.makedirs(output_dir)
                    except FileExistsError:
                        pass

                    expected_path = output_dir + '/trace.arff'
                    if os.path.isfile(expected_path):
                        print("Task already finished: %d exclude %s (%s) (path: %s)" % (task_id, exclude_param, str(value), expected_path))
                        continue

                    lock_file = fasteners.InterProcessLock(output_dir + '/tmp.lock')
                    obtained_lock = lock_file.acquire(blocking=False)

                    try:
                        if not obtained_lock:
                            # this means some other process already is working
                            print("Task already in progress: %d exclude %s (%s)" %(task_id, exclude_param, str(value)))
                            continue

                        optimizer = RandomizedSearchCV(estimator=pipeline,
                                                       param_distributions=param_distributions,
                                                       n_iter=args.n_iters,
                                                       random_state=1,
                                                       n_jobs=-1)
                        optimizer.set_params(**{'estimator__' + exclude_param: value})
                        # print(optimizer)

                        run = openml.runs.run_model_on_task(task, optimizer)
                        score = run.get_metric_fn(sklearn.metrics.accuracy_score)

                        print('%s [SCORE] Data: %s; Accuracy: %0.2f' %(openmlpimp.utils.get_time(), task.get_dataset().name, score.mean()))

                        trace_arff = arff.dumps(run._generate_trace_arff_dict())
                        run_xml = run._create_description_xml()
                        predictions_arff = arff.dumps(run._generate_arff_dict())

                        with open(output_dir + '/trace.arff', 'w') as f:
                            f.write(trace_arff)
                        with open(output_dir + '/run.xml', 'w') as f:
                            f.write(run_xml)
                        with open(output_dir + '/predictions.arff', 'w') as f:
                            f.write(predictions_arff)
                    finally:
                        if obtained_lock:
                            lock_file.release()
        else:
            raise ValueError('unknown optimizer.')
