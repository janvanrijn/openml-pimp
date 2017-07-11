import openml
import openmlpimp

from openml.exceptions import OpenMLServerException
from openml.tasks.functions import _list_tasks

from smac.tae.execute_ta_run import StatusType


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


def obtain_runhistory_and_configspace(flow_id, task_id, keyfield='parameter_name'):
    evaluations = openml.evaluations.list_evaluations("predictive_accuracy", flow=[flow_id], task=[task_id])
    setups = openml.setups.list_setups(flow=flow_id)

    data = []
    configs = {}
    applicable_setups = set()
    for run_id in evaluations.keys():
        cost = 1.0 - evaluations[run_id].value
        runtime = 0.0 # not easily accessible
        status = {"__enum__": str(StatusType.SUCCESS) }
        additional = {}
        performance = [cost, runtime, status, additional]

        config_id = evaluations[run_id].setup_id
        applicable_setups.add(evaluations[run_id].setup_id)
        instance = openml.config.server + "task/" + str(task_id)
        seed = 1 # not relevant
        run = [config_id, instance, seed]

        data.append([run, performance])

    for setup_id in applicable_setups:
        config = {}
        for param_id in setups[setup_id].parameters:
            name = getattr(setups[setup_id].parameters[param_id], keyfield)
            value = openml.flows.flow_to_sklearn(setups[setup_id].parameters[param_id].value)
            # TODO: hack
            if isinstance(value, bool):
                value = str(value)
            config[name] = value
        configs[setup_id] = config

    relevant_setups = {k: setups[k] for k in applicable_setups}
    config_space, constants = openmlpimp.utils.setups_to_configspace(relevant_setups, keyfield=keyfield)

    # remove the constants from runhistory TODO: make optional
    for config_id in configs:
        for constant in constants:
            configs[config_id].pop(constant, None)

    run_history = {"data": data, "configs": configs}
    return run_history, config_space
