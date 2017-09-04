
import argparse
import collections
import numpy as np
import openml
import openmlpimp


def parse_args():
  parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
  parser.add_argument('--study_id', type=int, default=14, help='the tag to obtain the tasks from')
  parser.add_argument('--flow_id', type=int, default=6969, help='the classifier to execute')  # 6952, 6969, 6970
  parser.add_argument('--fixed_parameter', type=str, default=None, help='parameter to split out')
  return parser.parse_args()


args = parse_args()

study = openml.study.get_study(args.study_id)
task_ids = {task_id: 0 for task_id in study.tasks}

offset = 0
limit = 10000
setups = set()
while True:
    runs = openml.runs.list_runs(flow=[args.flow_id], task=study.tasks, size=limit, offset=offset)
    for run_id, run in runs.items():
        task_id = run['task_id']
        task_ids[task_id] += 1
        setups.add(run['setup_id'])
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

if args.fixed_parameter:
    setups = openmlpimp.utils.obtain_setups_by_setup_id(list(setups), args.flow_id)

    values = collections.defaultdict(dict)
    for id, setup in setups.items():

        for param_id, param in setup.parameters.items():
            if param.parameter_name == args.fixed_parameter:
                values[param.parameter_name] += 1
print(values)