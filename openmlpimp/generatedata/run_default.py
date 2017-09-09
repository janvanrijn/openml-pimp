import argparse
import fasteners
import json
import openml
import openmlpimp
import os
import sklearn

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                       'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                       'qda', 'random_forest', 'sgd']
    all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest', 'sgd']
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/defaults')
    parser.add_argument('--study_id', type=str, default=14, help='the tag to obtain the tasks from')
    parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--classifier', type=str, choices=all_classifiers, default='random_forest', help='the classifier to execute')
    parser.add_argument('--fixed_parameters', type=json.loads, default=None, help='fixed parameters')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    study = openml.study.get_study(args.study_id)

    output_save_folder_suffix = openmlpimp.utils.fixed_parameters_to_suffix(args.fixed_parameters)

    for task_id in study.tasks:
        task = openml.tasks.get_task(task_id)

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

        if args.fixed_parameters is not None:
            fixed_param_prefix = {"classifier__" + param: value for param, value in args.fixed_parameters.items()}
            required_params.update(fixed_param_prefix)
        pipe.set_params(**required_params)
        output_dir = args.output_dir + '/' + args.classifier + output_save_folder_suffix + '/' + str(task_id)

        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass

        expected_path = output_dir + '/run.xml'
        if os.path.isfile(expected_path):
            print("Task already finished: %d (%s)" % (task_id, expected_path))
            continue

        lock_file = fasteners.InterProcessLock(output_dir + '/tmp.lock')
        obtained_lock = lock_file.acquire(blocking=False)
        try:
            if not obtained_lock:
                # this means some other process already is working
                print("Task already in progress: %d" %(task_id))
                continue

            try:
                run = openmlpimp.utils.do_run(task, pipe, output_dir, True, True)
            except Exception as e:
                expected_text = 'Run already exists in server. Run id(s): {'
                if e.message.startswith(expected_text):
                    relevant = e.message[len(expected_text):-1]
                    run_ids = relevant.split(', ')
                    run = openml.runs.get_run(int(run_ids[0]))
                else:
                    raise e

            score = run.get_metric_fn(sklearn.metrics.accuracy_score)

            print('%s [SCORE] Data: %s; Accuracy: %0.2f' % (openmlpimp.utils.get_time(), task.get_dataset().name, score.mean()))

        finally:
            if obtained_lock:
                lock_file.release()
