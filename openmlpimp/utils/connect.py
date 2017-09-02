import collections
import openml
import openmlpimp

import os
import json

from openml.exceptions import OpenMLServerException

from ConfigSpace.io.pcs_new import write


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


def obtain_setups_by_setup_id(setup_ids, flow):
    # because urls can be a bitch
    setups = {}
    offset = 0
    limit = 100
    while offset < len(setup_ids):
        setups_batch = openml.setups.list_setups(setup=setup_ids[offset:offset+limit], flow=flow)
        for setup_id in setups_batch.keys():
            setups[setup_id] = setups_batch[setup_id]

        offset += limit
        if len(setups_batch) < limit:
            break
    if set(setup_ids) != set(setups.keys()):
        raise ValueError()
    return setups


def obtain_all_evaluations(**kwargs):
    evaluations = {}
    offset = 0
    limit = 1000
    while True:
        evaluations_batch = openml.evaluations.list_evaluations(**kwargs, offset=offset, size=limit)
        for run_id in evaluations_batch.keys():
            evaluations[run_id] = evaluations_batch[run_id]

        offset += limit
        if len(evaluations_batch) < limit:
            break
    return evaluations


def obtain_all_setups(**kwargs):
    setups = {}
    offset = 0
    limit = 1000
    while True:
        setups_batch = openml.setups.list_setups(**kwargs, offset=offset, size=limit)
        for setup_id in setups_batch.keys():
            setups[setup_id] = setups_batch[setup_id]

        offset += limit
        if len(setups_batch) < limit:
            break
    return setups


def setup_complies_to_fixed_parameters(setup, keyfield, fixed_parameters):
    # tests whether a setup has the right values that are requisted by fixed parameters
    if fixed_parameters is None or len(fixed_parameters) == 0:
        return True
    setup_parameters = {getattr(setup.parameters[param_id], keyfield): setup.parameters[param_id].value for param_id in
                        setup.parameters}
    for parameter in fixed_parameters.keys():
        if parameter not in setup_parameters.keys():
            raise ValueError('Required parameter %s not in setup parameter for setup %d' % (parameter, setup.setup_id))
        value_online = openml.flows.flow_to_sklearn(setup_parameters[parameter])
        value_request = fixed_parameters[parameter]
        if value_online != value_request:
            return False
    return True


def obtain_setups(flow_id, setup_ids, keyfield, fixed_parameters):
    setups = {}
    offset = 0
    limit  = 500
    setup_ids = list(setup_ids)
    while True:
        setups_batch = openml.setups.list_setups(flow=flow_id, setup=setup_ids[offset:offset+limit], offset=offset)
        if fixed_parameters is None:
            setups.update(setups_batch)
        else:
            for setup_id in setups_batch.keys():
                if setup_complies_to_fixed_parameters(setups_batch[setup_id], keyfield, fixed_parameters):
                    setups[setup_id] = setups_batch[setup_id]

        offset += limit
        if len(setups_batch) < limit:
            break
    return setups


def obtain_runhistory_and_configspace(flow_id, task_id,
                                      model_type,
                                      keyfield='parameter_name',
                                      required_setups=None,
                                      fixed_parameters=None,
                                      ignore_parameters=None,
                                      reverse=False):
    from smac.tae.execute_ta_run import StatusType

    config_space = openmlpimp.utils.get_config_space_casualnames(model_type, fixed_parameters)
    valid_hyperparameters = config_space._hyperparameters.keys()

    evaluations = obtain_all_evaluations(function="predictive_accuracy", flow=[flow_id], task=[task_id])
    setup_ids = set()
    for run_id in evaluations.keys():
        setup_ids.add(evaluations[run_id].setup_id)

    if required_setups is not None:
        if len(setup_ids) < required_setups:
            raise ValueError('Not enough (evaluated) setups found on OpenML. Found %d; required: %d' %(len(setup_ids), required_setups))

    setups = obtain_setups(flow_id, setup_ids, keyfield, fixed_parameters)
    print('Setup count; before %d after %d' %(len(setup_ids), len(setups)))
    setup_ids = set(setups.keys())

    # filter again ..
    if required_setups is not None:
        if len(setup_ids) < required_setups:
            raise ValueError('Not enough (evaluated) setups left after filtering. Got %d; required: %d' %(len(setup_ids), required_setups))

    data = []
    configs = {}
    applicable_setups = set()
    for run_id in evaluations.keys():
        cost = evaluations[run_id].value
        runtime = 0.0 # not easily accessible
        status = {"__enum__": str(StatusType.SUCCESS)}
        additional = {}
        performance = [cost, runtime, status, additional]

        config_id = evaluations[run_id].setup_id
        instance = openml.config.server + "task/" + str(task_id)
        seed = 1  # not relevant
        run = [config_id, instance, seed]

        if config_id in setup_ids:
            applicable_setups.add(config_id)
            data.append([run, performance])

    # filter "constant" parameters by value
    param_values = collections.defaultdict(set)
    for setup_id in applicable_setups:
        for param_id in setups[setup_id].parameters:
            name = getattr(setups[setup_id].parameters[param_id], keyfield)
            value = openml.flows.flow_to_sklearn(setups[setup_id].parameters[param_id].value)
            if name in valid_hyperparameters:
                param_values[name].add(value)

    one_value_params = set()
    for hyperparameter, values in param_values.items():
        if len(values) == 1:
            one_value_params.add(hyperparameter)

    ignore_params = set()
    ignore_params.update(fixed_parameters)
    ignore_params.update(one_value_params)

    # this should update it ..
    config_space = openmlpimp.utils.get_config_space_casualnames(model_type, ignore_params)
    valid_hyperparameters = config_space._hyperparameters.keys()

    for setup_id in applicable_setups:
        config = {}
        for param_id in setups[setup_id].parameters:
            name = getattr(setups[setup_id].parameters[param_id], keyfield)
            value = openml.flows.flow_to_sklearn(setups[setup_id].parameters[param_id].value)
            if ignore_parameters is not None and name in ignore_parameters:
                continue
            if fixed_parameters is not None and name in fixed_parameters:
                continue
            if name not in valid_hyperparameters:
                continue
            # TODO: hack
            if isinstance(value, bool):
                value = str(value)
            config[name] = value
        configs[setup_id] = config

    run_history = {"data": data, "configs": configs}

    if reverse:
        openmlpimp.utils.reverse_runhistory(run_history)

    return run_history, config_space


def cache_runhistory_configspace(save_folder, flow_id, task_id, model_type, reverse, args):
    if args.fixed_parameters:
        save_folder_suffix = [param + '_' + value for param, value in args.fixed_parameters.items()]
        save_folder_suffix = '/' + '__'.join(save_folder_suffix)
    else:
        save_folder_suffix = '/vanilla'

    runhistory_path = save_folder + save_folder_suffix + '/runhistory.json'
    configspace_path = save_folder + save_folder_suffix + '/config_space.pcs'
    print(runhistory_path, configspace_path)

    if not os.path.isfile(runhistory_path) or not os.path.isfile(configspace_path):
        runhistory, configspace = openmlpimp.utils.obtain_runhistory_and_configspace(flow_id, task_id, model_type,
                                                                                     required_setups=args.required_setups,
                                                                                     fixed_parameters=args.fixed_parameters,
                                                                                     ignore_parameters=args.ignore_parameters,
                                                                                     reverse=reverse)

        try: os.makedirs(save_folder + save_folder_suffix)
        except FileExistsError: pass

        with open(runhistory_path, 'w') as outfile:
            json.dump(runhistory, outfile, indent=2)

        with open(configspace_path, 'w') as outfile:
            outfile.write(write(configspace))
    else:
        print('[Obtained from cache]')

    # now the files are guaranteed to exists
    return runhistory_path, configspace_path
