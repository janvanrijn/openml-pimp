
import argparse
import openml
import openmlpimp
import random
import traceback

from openml.exceptions import OpenMLServerException
from collections import OrderedDict

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
  parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
  parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
  parser.add_argument('--classifier', type=str, choices=all_classifiers, default='decision_tree', help='the classifier to execute')

  return parser.parse_args()


def get_probability_fn(configuration_space, all_task_ids):
    classifier = openmlpimp.utils.obtain_classifier(configuration_space, None)
    flow = openml.flows.sklearn_to_flow(classifier)
    flow_id = openml.flows.flow_exists(flow.name, flow.external_version)

    # obtain task counts
    task_counts = {}
    if flow_id: task_counts = openmlpimp.utils.task_counts(flow_id)

    # add tasks with count 0
    for task_id in all_task_ids:
        if task_id not in task_counts:
            task_counts[task_id] = 0

    max_value = 0
    if len(task_counts) > 0: max_value = max(1, max(task_counts.values()))

    # invert
    probability_fn = {}
    for task_id in task_counts:
        probability_fn[task_id] = max_value - task_counts[task_id]

     # sort (because why not)
    return OrderedDict(sorted(probability_fn.items()))

args = parse_args()
openml.config.apikey = args.openml_apikey
if args.openml_server:
    openml.config.server = args.openml_server

all_tasks = openml.tasks.list_tasks(tag=args.openml_tag)
all_task_ids = set(all_tasks.keys())


configuration_space = get_configuration_space(
    info={'task': autosklearn.constants.MULTICLASS_CLASSIFICATION, 'is_sparse': 0},
    include_estimators=[args.classifier],
    include_preprocessors=['no_preprocessing'])


weighted_probabilities = get_probability_fn(configuration_space, all_task_ids)

for i in range(args.n_executions):
    try:

        # sample a weighted random task
        task_id = random.choice([val for val, cnt in weighted_probabilities.items() for i in range(cnt)])
        # download task
        task = openml.tasks.get_task(task_id)
        indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])

        classifier = openmlpimp.utils.obtain_classifier(configuration_space, indices)

        # invoke OpenML run
        run = openml.runs.run_model_on_task(task, classifier)
        run.tags.append('openml-pimp')

        # and publish it
        run.publish()
        print(run.run_id)
    except ValueError as e:
        traceback.print_exc()
    except TypeError as e:
        traceback.print_exc()
    except OpenMLServerException as e:
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
    except:
        print('Unexpected error! ')