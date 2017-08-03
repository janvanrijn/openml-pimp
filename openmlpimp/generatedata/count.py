
import argparse
import openml
import numpy as np

def parse_args():
  parser = argparse.ArgumentParser(description = 'Generate data for openml-pimp project')
  parser.add_argument('--n_executions', type=int,  default=100, help='number of runs to be executed. ')
  parser.add_argument('--openml_tag', type=str, required=True, default=None, help='the tag to obtain the tasks from')
  parser.add_argument('--openml_apikey', type=str, required=True, default=None, help='the apikey to authenticate to OpenML')
  parser.add_argument('--flow_id', type=int, default=6952, help='the classifier to execute') # 6952, 6969, 6970

  return parser.parse_args()

args = parse_args()
openml.config.apikey = args.openml_apikey

task_ids = {}
offset = 0
limit = 10000
while True:
    runs = openml.runs.list_runs(flow=[args.flow_id], size=limit, offset=offset)
    for run_id, run in runs.items():
        task_id = run['task_id']
        if task_id not in task_ids:
            task_ids[task_id] = 0
        task_ids[task_id] += 1
    if len(runs) < limit:
        break
    else:
        offset += limit
print(args.flow_id)
print(task_ids)
print("tasks", len(task_ids))
print("sum", sum(task_ids.values()))
print("min", min(task_ids.values()))
print("max", max(task_ids.values()))
print("std", np.array(list(task_ids.values())).std())
