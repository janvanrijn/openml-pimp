import numpy as np
import json
import openml
import openmlpimp
import os
import pickle
from scipy.stats import gaussian_kde, rv_discrete, uniform

from collections import OrderedDict

from ConfigSpace.hyperparameters import CategoricalHyperparameter, NumericalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from openmlstudy14.distributions import loguniform, loguniform_int


class rv_discrete_wrapper(object):
    def __init__(self, param_name, X):
        self.param_name = param_name
        self.X_prime = OrderedDict()
        for value in X:
            if value not in self.X_prime:
                self.X_prime[value] = 0
            self.X_prime[value] = self.X_prime[value] + (1.0 / len(X))
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
        sample = self.distrib.rvs()
        value = list(self.X_prime.keys())[sample]
        if value in ['True', 'False']:
            return bool(value)
        elif self._is_castable_to(value, int):
            return int(value)
        elif self._is_castable_to(value, float):
            return float(value)
        else:
            return str(value)


class gaussian_kde_wrapper(object):
    def __init__(self, hyperparameter, param_name, X, log):
        self.param_name = param_name
        self.log = log
        self.const = False
        self.hyperparameter = hyperparameter
        try:
            if self.log:
                self.distrib = gaussian_kde(np.log2(X))
            else:
                self.distrib = gaussian_kde(X)
        except np.linalg.linalg.LinAlgError:
            self.distrib = rv_discrete(values=([X[0]], [1.0]))
            self.const = True

    def pdf(self, x):
        if self.const:
            raise ValueError('No pdf available for rv_sample')
        if self.log:
            return self.distrib.pdf(np.log2(x))
        else:
            return self.distrib.pdf(x)

    def rvs(self, *args, **kwargs):
        # assumes a samplesize of 1, for random search
        sample = self.distrib.resample(size=1)[0][0]
        if self.log:
            value = np.power(2, sample)
        else:
            value = sample
        if value < self.hyperparameter.lower:
            value = self.hyperparameter.lower
        elif value > self.hyperparameter.upper:
            value = self.hyperparameter.upper
        if isinstance(self.hyperparameter, UniformIntegerHyperparameter):
            value = int(value)
        return value


def cache_priors(cache_directory, study_id, flow_id, fixed_parameters):
    study = openml.study.get_study(study_id, 'tasks')
    if fixed_parameters is not None and len(fixed_parameters) > 0:
        print('No cache file for setups, will create one ... ')
        setups = openmlpimp.utils.obtain_all_setups(flow=flow_id)
    else:
        print('Obtained setups from cache')

    best_setupids = {}
    for task_id in study.tasks:
        print("task", task_id)
        runs = openml.evaluations.list_evaluations("predictive_accuracy", task=[task_id], flow=[flow_id])
        best_score = 0.0
        for run in runs.values():
            score = run.value
            if score > best_score and (fixed_parameters is None or len(fixed_parameters) == 0):
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

    holdout : int
        OpenML task id to not involve in the sampling

    Returns
    -------
    X : dict[str, list[mixed]]
        Mapping from hyperparameter name to a list of the best values.
    """
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
        for param_name, parameter in hyperparameters.items():
            param = setups[setup_id].parameters[paramname_paramidx[param_name]]
            if isinstance(parameter, NumericalHyperparameter):
                X[param_name].append(float(param.value))
            elif isinstance(parameter, CategoricalHyperparameter):
                X[param_name].append(json.loads(param.value))
            else:
                raise ValueError()

    for parameter in X:
        X[parameter] = np.array(X[parameter])
    return X


def get_prior_paramgrid(cache_directory, study_id, flow_id, hyperparameters, fixed_parameters, holdout=None):
    priors = obtain_priors(cache_directory, study_id, flow_id, hyperparameters, fixed_parameters, holdout)
    param_grid = dict()

    for parameter_name, prior in priors.items():
        if fixed_parameters is not None and parameter_name in fixed_parameters.keys():
            continue
        hyperparameter = hyperparameters[parameter_name]
        if isinstance(hyperparameter, CategoricalHyperparameter):
            param_grid[parameter_name] = rv_discrete_wrapper(parameter_name, prior)
        elif isinstance(hyperparameter, NumericalHyperparameter):
            param_grid[parameter_name] = gaussian_kde_wrapper(hyperparameter, parameter_name, prior, hyperparameter.log)
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

                param_grid[param_name] = uniform(loc=hyperparameter.lower, scale=hyperparameter.upper)
        elif isinstance(hyperparameter, UniformIntegerHyperparameter):
            if hyperparameter.log:
                param_grid[param_name] = loguniform_int(base=2, low=hyperparameter.lower, high=hyperparameter.upper)
            else:
                param_grid[param_name] = list(range(hyperparameter.lower, hyperparameter.upper))
        else:
            raise ValueError()
    return param_grid
