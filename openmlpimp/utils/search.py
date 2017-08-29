import collections
import numpy as np
import openmlpimp

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from scipy.stats import gaussian_kde
from sklearn.model_selection._search import BaseSearchCV


class KdeSampler(object):
    def __init__(self, param_priors, param_hyperparameters, n_iter):
        self.hyperparameters = collections.OrderedDict(sorted(param_hyperparameters.items()))
        self.n_iter = n_iter
        self.param_index = []
        self.distributions = {}

        kde_data = None
        for name, hyperparameter in param_hyperparameters.items():
            if isinstance(hyperparameter, UniformFloatHyperparameter):
                self.param_index.append(name)
                data = np.array(param_priors[name])
                if hyperparameter.log:
                    data = np.log2(data)
                if kde_data is None:
                    kde_data = np.reshape(np.array(data), (1, len(data)))
                else:
                    reshaped = np.reshape(np.array(data), (1, len(data)))
                    kde_data = np.concatenate((kde_data, reshaped), axis=0)
            elif isinstance(hyperparameter, UniformIntegerHyperparameter):
                raise ValueError('UniformIntegerHyperparameter not yet implemented:', name)
            elif isinstance(hyperparameter, CategoricalHyperparameter):
                self.distributions[name] = openmlpimp.utils.rv_discrete_wrapper(name, param_priors[name])
            else:
                raise ValueError()
        if len(self.param_index) < 2:
            raise ValueError('Need at least 2 float hyperparameters')
        self.kde = gaussian_kde(kde_data)

    def __iter__(self):
        # check if all distributions are given as lists
        # in this case we want to sample without replacement

        # Always sort the keys of a dictionary, for reproducibility
        for _ in range(self.n_iter):
            params = dict()
            sampled = self.kde.resample(size=1)
            for name, hyperparameter in self.hyperparameters.items():
                if isinstance(hyperparameter, UniformFloatHyperparameter):
                    value = sampled[self.param_index.index(name)][0]
                    if hyperparameter.log:
                        value = np.power(2, value)
                    params[name] = value
                elif isinstance(hyperparameter, UniformIntegerHyperparameter):
                    raise ValueError()
                elif isinstance(hyperparameter, CategoricalHyperparameter):
                    params[name] = self.distributions[name].rvs()
                else:
                    raise ValueError()

            yield params

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter


class MultivariateKdeSearch(BaseSearchCV):
    def __init__(self, estimator, param_priors, param_hyperparameters, n_iter=50,
                 scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=True):
        self.param_priors = param_priors
        self.param_distributions = param_priors # TODO: hack for OpenML plugin
        self.param_hyperparameters = param_hyperparameters
        self.n_iter = n_iter
        self.random_state = random_state
        super(MultivariateKdeSearch, self).__init__(
             estimator=estimator, scoring=scoring, fit_params=fit_params,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return KdeSampler(self.param_priors, self.param_hyperparameters, self.n_iter)

    def fit(self, X, y=None, groups=None):
        # compatibility function
        return self._fit(X, y, groups, self._get_param_iterator())