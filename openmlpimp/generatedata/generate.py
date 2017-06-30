
import argparse
import openml
import openmlpimp
import random

import autosklearn.constants
from autosklearn.util.pipeline import get_configuration_space


def parse_args():
  parser = argparse.ArgumentParser(description = 'Generate data for openml-pimp project')
  all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                     'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                     'qda', 'random_forest', 'sgd']
  all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest', 'sgd']
  parser.add_argument('--n_executions', type=int,  default=100, help='number of runs to be executed. ')
  parser.add_argument('--openml_tag', type=str, required=True, default=None, help='the tag to obtain the tasks from')
  parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
  parser.add_argument('--classifier', type=str, choices=all_classifiers, default='decision_tree', help='the classifier to execute')

  return parser.parse_args()


def obtain_classifier(configuration_space):
    for i in range(5):
        try:
            configuration = configuration_space.sample_configuration(1)
            classifier = openmlpimp.utils.config_to_classifier(configuration)
            return classifier
        except ValueError:
            # sometimes a classifier is not valid. TODO, check this
            pass

args = parse_args()
openml.config.apikey = args.openml_apikey

all_tasks = openml.tasks.list_tasks(tag=args.openml_tag)
all_task_ids = set(all_tasks.keys())

configuration_space = get_configuration_space(
    info={'task': autosklearn.constants.MULTICLASS_CLASSIFICATION, 'is_sparse': 0},
    include_estimators=[args.classifier],
    include_preprocessors=['no_preprocessing'])


for i in range(args.n_executions):
    try:
        classifier = obtain_classifier(configuration_space)

        # sample task
        task_id = random.sample(all_task_ids, 1)[0]

        # download task
        task = openml.tasks.get_task(task_id)


        # invoke OpenML run
        run = openml.runs.run_model_on_task(task, classifier)
        run.tags.append('openml-pimp')

        # and publish it
        run_prime = run.publish()
        print(run_prime.run_id)
    except ValueError:
        # anything can go wrong. prevent this
        pass