import argparse
import collections
import numpy as np
import openml
import openmlpimp


def parse_args():
  parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
  parser.add_argument('--n_executions', type=int,  default=100, help='number of runs to be executed. ')
  parser.add_argument('--study_id', type=int, default=14, help='the tag to obtain the tasks from')
  parser.add_argument('--flow_id', type=int, default=6952, help='the classifier to execute') # 6952, 6969, 6970
  return parser.parse_args()


def evals_to_list(evals):
    return np.array([eval.value for eval in evals.values()])

args = parse_args()
study = openml.study.get_study(args.study_id)


for task_id in study.tasks:
    name = openml.tasks.get_task(task_id).get_dataset().name
    accuracy = evals_to_list(openml.evaluations.list_evaluations("predictive_accuracy", flow=[args.flow_id], task=[task_id], size=500))
    # auroc = evals_to_list(openml.evaluations.list_evaluations("area_under_roc_curve", flow=args.flow_id, task=task_id, size=500))
    # runtime = evals_to_list(openml.evaluations.list_evaluations("usercpu_time_millis", flow=[args.flow_id], task=[task_id], size=500))

    print('%7d %20s %7f %7f %7f' %(task_id, name, np.min(accuracy), np.median(accuracy), np.max(accuracy)))
