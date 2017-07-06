import openml
import openmlpimp

from openml.exceptions import OpenMLServerException


def get_flow_id(configuration_space):
    # for getting the flow id of the appropriate model
    classifier = openmlpimp.utils.obtain_classifier(configuration_space)
    flow = openml.flows.sklearn_to_flow(classifier)
    return openml.flows.flow_exists(flow.name, flow.external_version)


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