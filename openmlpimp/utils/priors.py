import numpy as np
import openml
import openmlpimp
import os
import pickle


def cache_priors(cache_directory, study_id, flow_id, fixed_parameters):
    study = openml.study.get_study(study_id, 'tasks')
    if fixed_parameters is not None and len(fixed_parameters) > 0:
        setups = openmlpimp.utils.obtain_all_setups(flow=flow_id)

    best_setupids = {}
    for task_id in study.tasks:
        print("task", task_id)
        runs = openml.evaluations.list_evaluations("predictive_accuracy", task=[task_id], flow=[flow_id])
        best_score = 0.0
        for run in runs.values():
            score = run.value
            if run.setup_id not in setups:
                raise ValueError()
            if score > best_score and len(fixed_parameters) == 0:
                best_score = score
                best_setupids[task_id] = run.setup_id
            elif score > best_score and openmlpimp.utils.setup_complies_to_fixed_parameters(setups[run.setup_id],
                                                                                            'parameter_name',
                                                                                            fixed_parameters):
                best_score = score
                best_setupids[task_id] = run.setup_id

                # if len(best_setupids) > 10: break
    try:
        os.makedirs(cache_directory)
    except FileExistsError:
        pass

    with open(cache_directory + '/best_setup_per_task.pkl', 'wb') as f:
        pickle.dump(best_setupids, f, pickle.HIGHEST_PROTOCOL)


def obtain_priors(cache_directory, study_id, flow_id, hyperparameters, fixed_parameters, holdout=None):
    filename = cache_directory + '/best_setup_per_task.pkl'
    if not os.path.isfile(filename):
        cache_priors(cache_directory, study_id, flow_id, fixed_parameters)

    with open(filename, 'rb') as f:
        best_setupids = pickle.load(f)

    X = {parameter: list() for parameter in hyperparameters.keys()}
    setups = openml.setups.list_setups(setup=list(best_setupids.values()), flow=flow_id)

    for task_id, setup_id in best_setupids.items():
        if task_id == holdout:
            print('Holdout task %d' %task_id)
            continue
        paramname_paramidx = {param.parameter_name: idx for idx, param in setups[setup_id].parameters.items()}
        for parameter in hyperparameters.keys():
            param = setups[setup_id].parameters[paramname_paramidx[parameter]]
            X[parameter].append(float(param.value))

    for parameter in X:
        X[parameter] = np.array(X[parameter])
    return X
