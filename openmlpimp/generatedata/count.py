
import argparse
import openml
import openmlpimp
import numpy as np

import autosklearn.constants
from autosklearn.util.pipeline import get_configuration_space


def parse_args():
  parser = argparse.ArgumentParser(description = 'Generate data for openml-pimp project')
  all_classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting',
                     'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive',
                     'qda', 'random_forest', 'sgd']
  all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest']
  parser.add_argument('--n_executions', type=int,  default=100, help='number of runs to be executed. ')
  parser.add_argument('--openml_tag', type=str, required=True, default=None, help='the tag to obtain the tasks from')
  parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
  parser.add_argument('--classifier', type=str, choices=all_classifiers, default='decision_tree', help='the classifier to execute')

  return parser.parse_args()

args = parse_args()
openml.config.apikey = args.openml_apikey


#adaboost 6970
#random forest 6969
flow_id = [6969]

task_ids = {}
offset = 0
limit = 10000
while True:
    runs = openml.runs.list_runs(flow=flow_id, size=limit, offset=offset)
    for run_id, run in runs.items():
        task_id = run['task_id']
        if task_id not in task_ids:
            task_ids[task_id] = 0
        task_ids[task_id] += 1
    if len(runs) < limit:
        break
    else:
        offset += limit
print(task_ids)
print("tasks", len(task_ids))
print("sum", sum(task_ids.values()))
print("min", min(task_ids.values()))
print("max", max(task_ids.values()))
print("std", np.array(list(task_ids.values())).std())