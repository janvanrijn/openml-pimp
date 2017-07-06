import openml

from openml.exceptions import OpenMLServerException
from openml.tasks.functions import _list_tasks


def list_tasks(task_type_id=None, offset=None, size=None, tag=None, nr_instances=None):
    api_call = "task/list"
    if task_type_id is not None:
        api_call += "/type/%d" % int(task_type_id)

    if offset is not None:
        api_call += "/offset/%d" % int(offset)

    if size is not None:
        api_call += "/limit/%d" % int(size)

    if tag is not None:
        api_call += "/tag/%s" % tag

    if nr_instances is not None:
        api_call += "/number_instances/%s" %nr_instances

    return _list_tasks(api_call)


def task_counts(flow_id):
    task_ids = {}
    offset = 0
    limit = 10000
    while True:
        try:
            runs = openml.runs.list_runs(flow=[flow_id], size=limit, offset=offset)
        except OpenMLServerException:
            runs = {}

        for run_id, run in runs.items():
            task_id = run['task_id']
            if task_id not in task_ids:
                task_ids[task_id] = 0
            task_ids[task_id] += 1
        if len(runs) < limit:
            break
        else:
            offset += limit
    return task_ids