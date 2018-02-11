import collections
import numpy as np
import json
import openml
import openmlpimp
import operator
import os
import pickle
import warnings

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde, rv_discrete, uniform, randint
from ConfigSpace.hyperparameters import CategoricalHyperparameter, NumericalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from openmlstudy14.distributions import loguniform, loguniform_int


class rv_discrete_wrapper(object):
    def __init__(self, param_name, X):
        self.param_name = param_name
        self.X_prime = collections.OrderedDict()
        for value in X:
            if value not in self.X_prime:
                self.X_prime[value] = 0
            self.X_prime[value] += (1.0 / len(X))
        self.distrib = rv_discrete(values=(list(range(len(self.X_prime))), list(self.X_prime.values())))

    @staticmethod
    def _is_castable_to(value, type):
        try:
            type(value)
            return True
        except ValueError:
            return False

    def rvs(self, *args, **kwargs):
        # assumes a samplesize of 1, for random search
        sample = self.distrib.rvs(*args, **kwargs)
        value = list(self.X_prime.keys())[sample]

        if value in ['True', 'False']:
            return bool(value)
        elif self._is_castable_to(value, int):
            return int(value)
        elif self._is_castable_to(value, float):
            return float(value)
        else:
            return str(value)


# class empirical_distribution_wrapper(object):
#
#     def __init__(self, hyperparameter, priors):
#         self.hyperparameter = hyperparameter
#         self.distrib = EmpiricalDistribution(priors)
#
#     def rvs(self, *args, **kwargs):
#         # assumes a samplesize of 1, for random search
#         sample = self.distrib.rvs(1)[0]
#         if isinstance(self.hyperparameter, UniformIntegerHyperparameter):
#             return int(sample)
#         return sample


class gaussian_kde_wrapper(object):
    def __init__(self, hyperparameter, param_name, data, oob_strategy='resample', bandwith=0.4):
        if oob_strategy not in ['resample', 'round', 'ignore']:
            raise ValueError()
        self.oob_strategy = oob_strategy
        self.param_name = param_name
        self.hyperparameter = hyperparameter
        reshaped = np.reshape(data, (len(data), 1))

        if self.hyperparameter.log:
            if isinstance(self.hyperparameter, UniformIntegerHyperparameter):
                # self.probabilities = {val: self.distrib.pdf(np.log2(val)) for val in range(self.hyperparameter.lower, self.hyperparameter.upper)}
                raise ValueError('Log Integer hyperparameter not supported: %s' %param_name)
            # self.distrib = gaussian_kde(np.log2(data))
            # self.distrib = KernelDensity(kernel='gaussian').fit(np.log2(np.reshape(data, (len(data), 1))))
            self.distrib = KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(np.log2(reshaped))
        else:
            # self.distrib = gaussian_kde(data)
            self.distrib = KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(reshaped)
        pass

    def pdf(self, x):
        x = np.reshape(x, (len(x), 1))
        if self.hyperparameter.log:
            x = np.log2(x)
        log_dens = self.distrib.score_samples(x)
        return np.exp(log_dens)

    def rvs(self, *args, **kwargs):
        # assumes a samplesize of 1, for random search
        while True:
            sample = self.distrib.sample(n_samples=1, random_state=kwargs['random_state'])[0][0]
            if self.hyperparameter.log:
                value = np.power(2, sample)
            else:
                value = sample
            if isinstance(self.hyperparameter, UniformIntegerHyperparameter):
                value = int(round(value))

            if self.hyperparameter.lower <= value <= self.hyperparameter.upper:
                return value
            elif self.oob_strategy == 'ignore':
                # TODO: hacky fail safe for some hyperparameters
                if hasattr(self.hyperparameter, 'lower_hard') and self.hyperparameter.lower_hard > value:
                    continue
                if hasattr(self.hyperparameter, 'upper_hard') and self.hyperparameter.upper_hard < value:
                    continue

                return value
            elif self.oob_strategy == 'round':
                if value < self.hyperparameter.lower:
                    return self.hyperparameter.lower
                elif value > self.hyperparameter.upper:
                    return self.hyperparameter.upper


def _get_best_setups(task_setup_scores, holdout, bestN, factor=4):
    task_setups = dict()
    for task, setup_scores in task_setup_scores.items():
        if (holdout is None or task not in holdout) and len(setup_scores) < bestN * factor:
            pass
            #raise Warning('Not enough setups for task %d. Need %d, expected at least %d, got %d' % (task, bestN, bestN * factor, len(setup_scores)))
        else:
            task_setups[task] = dict(sorted(setup_scores.items(), key=operator.itemgetter(1), reverse=True)[:bestN]).keys()
    return task_setups


def cache_setups(cache_directory, flow_id, bestN):
    try:
        os.makedirs(cache_directory)
    except FileExistsError:
        pass

    setups = openmlpimp.utils.obtain_all_setups(flow=flow_id)
    with open(cache_directory + '/setup_list_best%d.pkl' %bestN, 'wb') as f:
        pickle.dump(setups, f, pickle.HIGHEST_PROTOCOL)


def cache_task_setup_scores(cache_directory, study, flow_id, setups, fixed_parameters, hyperparameters, bestN):
    # print(setups.keys())
    task_setup_scores = collections.defaultdict(dict)
    for task_id in study.tasks:
        runs = openmlpimp.utils.obtain_all_evaluations(function="predictive_accuracy", task=[task_id], flow=[flow_id])
        for run in runs.values():
            if openmlpimp.utils.setup_complies_to_fixed_parameters(setups[run.setup_id], 'parameter_name', fixed_parameters):
                if openmlpimp.utils.setup_complies_to_config_space(setups[run.setup_id], keyfield='parameter_name', hyperparameters=hyperparameters):
                    task_setup_scores[task_id][run.setup_id] = run.value
    try:
        os.makedirs(cache_directory)
    except FileExistsError:
        pass

    task_setups = _get_best_setups(task_setup_scores, holdout=None, bestN=bestN * 5, factor=1)
    all_setup_ids = set()
    for setups in task_setups.values():
        all_setup_ids |= setups

    setups = openmlpimp.utils.obtain_setups_by_setup_id(setup_ids=list(all_setup_ids), flow=flow_id)
    missing = all_setup_ids - set(setups.keys())
    if len(missing) > 0:
        raise ValueError('Missing:', missing)

    with open(cache_directory + '/best_setup_per_task.pkl', 'wb') as f:
        pickle.dump(task_setup_scores, f, pickle.HIGHEST_PROTOCOL)


def obtain_priors(cache_directory, study_id, flow_id, hyperparameters, fixed_parameters, holdout, bestN):
    """
    Obtains the priors based on (almost) all tasks in an OpenML study

    Parameters
    -------
    cache_directory : str
        a directory on the filesystem to store and obtain the cache from

    study_id : int
        the study id to obtain the priors from

    flow id : int
        the flow id of the classifier

    hyperparameters : dict[str, ConfigSpace.Hyperparameter]
        dictionary mapping from parameter name to the ConfigSpace Hyperparameter object

    fixed_parameters : dict[str, str]
        maps from hyperparameter name to a value. Only setups are considered
        that have this hyperparameter set to this specific value

    holdout : list[int]
        OpenML task id to not involve in the sampling

    bestN : int
        from each task, take the N best setups.

    Returns
    -------
    X : dict[str, list[mixed]]
        Mapping from hyperparameter name to a list of the best values.
    """
    priors_cache_file = cache_directory + '/best_setup_per_task.pkl'
    setups_cache_file = cache_directory + '/setup_list_best%d.pkl' %bestN

    if not os.path.isfile(setups_cache_file):
        print('%s No cache file for setups (expected: %s), will create one ... ' %(openmlpimp.utils.get_time(), setups_cache_file))
        cache_setups(cache_directory, flow_id, bestN)
        print('%s Cache created. Available in: %s' %(openmlpimp.utils.get_time(), setups_cache_file))

    with open(setups_cache_file, 'rb') as f:
        setups = pickle.load(f)

    if not os.path.isfile(priors_cache_file):
        print('%s No cache file for task setup scores (expected: %s), will create one ... ' % (openmlpimp.utils.get_time(), priors_cache_file))
        study = openml.study.get_study(study_id, 'tasks')
        cache_task_setup_scores(cache_directory, study, flow_id, setups, fixed_parameters, hyperparameters, bestN)
        print('%s Cache created. Available in: %s' % (openmlpimp.utils.get_time(), priors_cache_file))

    with open(priors_cache_file, 'rb') as f:
        task_setup_scores = pickle.load(f)


    task_setups = _get_best_setups(task_setup_scores, holdout, bestN)
    all_setup_ids = set()
    for setup_id in task_setups.values():
        all_setup_ids |= setup_id

    if set(setups.keys()) != all_setup_ids:
        mismatch1 = all_setup_ids - set(setups.keys())
        mismatch2 = set(setups.keys()) - all_setup_ids
        if len(mismatch1) > 0:
            print(mismatch1) # TODO: JvR fix me
            print(mismatch2)
            raise ValueError('FIX ME. old serialization bug still active.')

    X = {parameter: list() for parameter in hyperparameters.keys()}

    for task_id, best_setups in task_setups.items():
        if holdout is not None and task_id in holdout:
            print('Holdout task %d' %task_id)
            continue

        for setup_id in best_setups:
            paramname_paramidx = {param.parameter_name: idx for idx, param in setups[setup_id].parameters.items()}
            for param_name, parameter in hyperparameters.items():
                param = setups[setup_id].parameters[paramname_paramidx[param_name]]
                if isinstance(parameter, NumericalHyperparameter):
                    try:
                        X[param_name].append(float(param.value))
                    except ValueError:
                        X[param_name].append(json.loads(param.value))
                elif isinstance(parameter, CategoricalHyperparameter):
                    X[param_name].append(json.loads(param.value))
                else:
                    raise ValueError()

    for parameter in X:
        if len(X[parameter]) == 0:
            raise ValueError('Did not obtain priors for task. ')
        X[parameter] = np.array(X[parameter])
    return X


def get_kde_paramgrid(cache_directory, study_id, flow_id, hyperparameters, fixed_parameters, holdout=None, bestN=1, oob_strategy='resample'):
    priors = obtain_priors(cache_directory, study_id, flow_id, hyperparameters, fixed_parameters, holdout, bestN)
    param_grid = dict()

    for parameter_name, prior in priors.items():
        if fixed_parameters is not None and parameter_name in fixed_parameters.keys():
            continue
        if all(x == prior[0] for x in prior):
            warnings.warn('Skipping Hyperparameter %s: All prior values equals (%s). ' %(parameter_name, prior[0]))
            continue
        hyperparameter = hyperparameters[parameter_name]
        if isinstance(hyperparameter, CategoricalHyperparameter):
            param_grid[parameter_name] = rv_discrete_wrapper(parameter_name, prior)
        elif isinstance(hyperparameter, NumericalHyperparameter):
            param_grid[parameter_name] = gaussian_kde_wrapper(hyperparameter, parameter_name, prior, oob_strategy)
        else:
            raise ValueError()
    return param_grid


def get_empericaldistribution_paramgrid(cache_directory, study_id, flow_id, hyperparameters, fixed_parameters, holdout=None, bestN=1):
    priors = obtain_priors(cache_directory, study_id, flow_id, hyperparameters, fixed_parameters, holdout, bestN)
    param_grid = dict()

    for parameter_name, prior in priors.items():
        if fixed_parameters is not None and parameter_name in fixed_parameters.keys():
            continue
        if all(x == prior[0] for x in prior):
            warnings.warn('Skipping Hyperparameter %s: All prior values equals (%s). ' %(parameter_name, prior[0]))
            continue
        hyperparameter = hyperparameters[parameter_name]
        if isinstance(hyperparameter, CategoricalHyperparameter):
            param_grid[parameter_name] = rv_discrete_wrapper(parameter_name, prior)
        elif isinstance(hyperparameter, NumericalHyperparameter):
            param_grid[parameter_name] = empirical_distribution_wrapper(hyperparameter, prior)
        else:
            raise ValueError()
    return param_grid


def get_uniform_paramgrid(hyperparameters, fixed_parameters):
    param_grid = dict()
    for param_name, hyperparameter in hyperparameters.items():
        if fixed_parameters is not None and param_name in fixed_parameters.keys():
            continue
        if isinstance(hyperparameter, CategoricalHyperparameter):
            all_values = hyperparameter.choices
            if all(item in ['True', 'False'] for item in all_values):
                all_values = [bool(item) for item in all_values]
            param_grid[param_name] = all_values
        elif isinstance(hyperparameter, UniformFloatHyperparameter):
            if hyperparameter.log:
                param_grid[param_name] = loguniform(base=2, low=hyperparameter.lower, high=hyperparameter.upper)
            else:
                param_grid[param_name] = uniform(loc=hyperparameter.lower, scale=hyperparameter.upper-hyperparameter.lower)
        elif isinstance(hyperparameter, UniformIntegerHyperparameter):
            if hyperparameter.log:
                param_grid[param_name] = loguniform_int(base=2, low=hyperparameter.lower, high=hyperparameter.upper)
            else:
                param_grid[param_name] = randint(low=hyperparameter.lower, high=hyperparameter.upper+1)
        else:
            raise ValueError()
    return param_grid
