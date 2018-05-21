import json
import numpy as np
import openml
import openmlpimp
import scipy
import sys

from openmlstudy14.distributions import loguniform, loguniform_gen
from collections import defaultdict, OrderedDict


def obtain_runids(task_ids, flow_id, classifier, param_templates):
    """
    Obtains relevant run ids from OpenML.

    Parameters
    -------
    task_ids : list[int]
        a list of the relevant task ids

    flow id : int
        the flow id of the optimizer

    classifier : str
        string representation of classifier (not OpenML based)
        for random forest: 'random_forest'

    param_templates : dict[str, dict[str, list]]
        maps from parameter name (sklearn representation) to param grid
        (which is a dict, mapping from parameter name to a list of values)

    Returns
    -------
    results : dict[str, dict[int, dict[mixed, list[ints]]]]
        a dict mapping from parameter name (sklearn representation) to a dict.
        This dict maps from an int (task id) to a dict
        This dict maps from a mixed value (the value of the excluded param) to a list.
        This list contains run ids (ideally 1, but accidently more).
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    setups = openmlpimp.utils.obtain_all_setups(flow=flow_id)

    for task_id in task_ids:
        print("task", task_id)
        try:
            runs = openml.runs.list_runs(task=[task_id], flow=[flow_id])
        except:
            print("runs None")
            continue

        for run_id in runs:
            setup_id = runs[run_id]['setup_id']
            if setup_id not in setups:
                # occurs when experiments are still running.
                sys.stderr.write('setup not available. (should not happen!) %d' %setup_id)
                setups[setup_id] = openml.setups.get_setup(setup_id)

            paramname_paramidx = {param.parameter_name: idx for idx, param in setups[setup_id].parameters.items()}

            for idx, parameter in setups[setup_id].parameters.items():
                if parameter.parameter_name == 'param_distributions':
                    param_grid = openml.flows.flow_to_sklearn(parameter.value)
                    if not isinstance(param_grid, dict):
                        continue
                    excluded_params = openmlpimp.utils.get_excluded_params(classifier, param_grid)
                    if len(excluded_params) > 1:
                        continue
                    excluded_param = list(excluded_params)[0]

                    excluded_param_idx = paramname_paramidx[excluded_param.split('__')[-1]]
                    excluded_param_openml = setups[setup_id].parameters[excluded_param_idx]
                    excluded_value = json.loads(excluded_param_openml.value)

                    # TODO: check if legal
                    # TODO: fixed parameters

                    for name, param_template in param_templates.items():
                        if param_template == param_grid:
                            results[name][task_id][excluded_value].append(run_id)
    return results


def obtain_parameters(classifier, fixed_parameters=None):
    return set(obtain_paramgrid(classifier, fixed_parameters=fixed_parameters).keys())


def obtain_parameter_combinations(classifier, num_params):
    if num_params != 2:
        raise ValueError('Not implemented yet')
    result = list()
    params = set(obtain_paramgrid(classifier).keys())
    for param1 in params:
        for param2 in params:
            if param1 == param2:
                continue
            result.append([param1, param2])
    return result


def get_excluded_params(classifier, param_grid):
    if not isinstance(param_grid, dict):
        raise ValueError()
    all_params = obtain_paramgrid(classifier).keys()
    return set(all_params - param_grid)


def get_param_values(classifier, parameter, fixed_parameters=None):
    param_grid = obtain_paramgrid(classifier, fixed_parameters=fixed_parameters)
    steps = 10
    if parameter not in param_grid:
        raise ValueError()
    grid = param_grid[parameter]
    if isinstance(grid, list):
        if len(grid) < steps:
            return grid
        min_val = min(grid)
        max_val = max(grid)
        dtype = float
        stepsize = np.ceil((max_val - min_val) / steps)
        if all(isinstance(item, int) for item in grid):
            dtype = int
        result = np.arange(min_val, max_val + 1, stepsize, dtype)
    elif hasattr(grid, 'dist') and isinstance(grid.dist, loguniform_gen):
        dtype = float
        result = grid.dist.logspace(steps)
    elif hasattr(grid, 'dist') and isinstance(grid.dist, scipy.stats._discrete_distns.randint_gen):
        dtype = int
        result = np.linspace(start=grid.dist.a, stop=grid.dist.b, num=steps, endpoint=True, dtype=dtype)
    elif hasattr(grid, 'dist') and isinstance(grid.dist, scipy.stats._continuous_distns.uniform_gen):
        dtype = float
        result = np.linspace(start=grid.dist.a, stop=grid.dist.b, num=steps, endpoint=True, dtype=dtype)
    else:
        raise ValueError('Illegal param grid: %s %s' %(classifier, parameter))

    return [dtype(val) for val in result]  # TODO: hacky


def obtain_paramgrid(classifier, exclude=None, reverse=False, fixed_parameters=None):
    if classifier == 'random_forest':
        param_grid = OrderedDict()
        param_grid['classifier__min_samples_leaf'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        param_grid['classifier__max_features'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        param_grid['classifier__bootstrap'] = [True, False]
        param_grid['classifier__min_samples_split'] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        param_grid['classifier__criterion'] = ['gini', 'entropy']
        param_grid['imputation__strategy'] = ['mean','median','most_frequent']
    elif classifier == 'adaboost':
        param_grid = dict()
        param_grid['classifier__n_estimators'] = scipy.stats.randint(low=50, high=500+1)
        param_grid['classifier__learning_rate'] = loguniform(base=2, low=10**-2, high=2)
        param_grid['classifier__algorithm'] = ['SAMME', 'SAMME.R']
        param_grid['classifier__base_estimator__max_depth'] = list(range(1, 10+1))
        param_grid['imputation__strategy'] = ['mean', 'median', 'most_frequent']
    elif classifier == 'libsvm_svc':
        if fixed_parameters['kernel'] == 'poly':
            param_grid = dict()
            param_grid['classifier__coef0'] = scipy.stats.uniform(loc=-1.0, scale=2.0)
            param_grid['classifier__degree'] = [1, 2, 3, 4]
            param_grid['classifier__gamma'] = loguniform(base=2, low=2 ** -15, high=8)
            param_grid['classifier__tol'] = loguniform(base=2, low=10 ** -5, high=0.1)
            param_grid['classifier__C'] = loguniform(base=2, low=2 ** -5, high=2 ** 15)
            param_grid['imputation__strategy'] = ['mean', 'median', 'most_frequent']
            param_grid['classifier__shrinking'] = [True, False]
        elif fixed_parameters['kernel'] == 'rbf':
            param_grid = dict()
            param_grid['classifier__gamma'] = loguniform(base=2, low=2 ** -15, high=8)
            param_grid['classifier__tol'] = loguniform(base=2, low=10 ** -5, high=0.1)
            param_grid['classifier__C'] = loguniform(base=2, low=2 ** -5, high=2 ** 15)
            param_grid['imputation__strategy'] = ['mean', 'median', 'most_frequent']
            param_grid['classifier__shrinking'] = [True, False]
            pass
        elif fixed_parameters['kernel'] == 'sigmoid':
            param_grid = dict()
            param_grid['classifier__gamma'] = loguniform(base=2, low=2 ** -15, high=8)
            param_grid['classifier__coef0'] = scipy.stats.uniform(loc=-1.0, scale=2.0)
            param_grid['classifier__C'] = loguniform(base=2, low=2 ** -5, high=2 ** 15)
            param_grid['classifier__tol'] = loguniform(base=2, low=10 ** -5, high=0.1)
            param_grid['imputation__strategy'] = ['mean', 'median', 'most_frequent']
            param_grid['classifier__shrinking'] = [True, False]
            pass
        else:
            raise ValueError()
    else:
        raise ValueError()

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        for exclude_param in exclude:
            if exclude_param not in param_grid.keys():
                raise ValueError()
            del param_grid[exclude_param]

    if reverse:
        return OrderedDict(reversed(list(param_grid.items())))
    else:
        return param_grid
