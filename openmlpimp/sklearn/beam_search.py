import numpy as np
import copy
import warnings
import collections

from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics.scorer import check_scoring


class BeamSampler(object):
    """In accordance with the Observer pattern """

    def __init__(self, param_distributions, beam_width, defaults):
        self.param_distributions = param_distributions
        self.beam_width = beam_width
        self.recent_results = collections.defaultdict(dict)
        self.param_order = []
        self.defaults = dict()

        for param, value in defaults.items():
            if param in self.param_distributions:
                if value in self.param_distributions[param]:
                    self.defaults[param] = value
                else:
                    self.defaults[param] = self.param_distributions[param][0]
                    warnings.warn('Parameter %s: Default value not on grid, picking instead %s' %(param, str(self.defaults[param])))

        if set(self.defaults.keys()) != set(self.param_distributions.keys()):
            raise ValueError('Param distributions and defaults do not agree. ')

        all_lists = np.all([not hasattr(v, "rvs") for v in self.param_distributions.values()])
        if not all_lists:
            raise ValueError('Currently only lists of parameter values are supported. ')
        if self.beam_width == 1:
            self.n_iter = sum([len(param_distributions[key]) for key in param_distributions])
        else:
            raise ValueError('Currently, only beam_width = 1 is supported. ')

    def __iter__(self):
        params = copy.deepcopy(self.defaults)
        for param, values in self.param_distributions.items():
            self.param_order.append(param)
            for value in values:
                if value == self.defaults[param] and len(self.param_order) > 1:
                    # we can skip results that were already ran
                    # (skip all params that match their default, unless we are optimizing param #1)
                    # just put it in the recent results
                    previous_param = self.param_order[-2]
                    previous_value = params[previous_param]
                    self.recent_results[param][self.defaults[param]] = self.recent_results[previous_param][previous_value]
                    continue
                params[param] = value
                yield copy.deepcopy(params)
            # now obtain the value of params that performed best and fix it (priority on later keys)
            params[param] = max(self.recent_results[param], key=lambda i: np.mean(self.recent_results[param][i]))

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter

    def update(self, params, score):
        if len(self.param_order) == 0:
            raise ValueError('param_order is None, call to update function unexpected. ')
        current_param = self.param_order[-1]
        current_value = params[current_param]
        if current_param not in self.recent_results:
            self.recent_results[current_param] = dict()
        if current_value not in self.recent_results[current_param]:
            self.recent_results[current_param][current_value] = []
        self.recent_results[current_param][current_value].append(score)


class ObservableScorer(object):
    """According to Observable pattern. """
    def __init__(self, original_scoring_fn):
        self.observers = []
        self.original_scoring_fn = original_scoring_fn

    def score(self, estimator, X, y):
        score = self.original_scoring_fn(estimator, X, y)
        self.update_observers(estimator.get_params(), score)
        return score

    def register(self, observer):
        if not observer in self.observers:
            self.observers.append(observer)

    def unregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def unregister_all(self):
        if self.observers:
            del self.observers[:]

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)


class BeamSearchCV(BaseSearchCV):

    def __init__(self, estimator, param_distributions, beam_width=1, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise'):
        if n_jobs > 1:
            raise ValueError('Multiprocessing not supported yet (please fix n_jobs to 1). ')
        self.param_distributions = param_distributions
        self.beam_width = beam_width
        self.defaults = estimator.get_params()

        # trick to communicate scores back to param grid.
        self.original_scoring = check_scoring(estimator, scoring)
        self.observable_scorer = ObservableScorer(self.original_scoring)
        scoring = self.observable_scorer.score

        super(BeamSearchCV, self).__init__(
             estimator=estimator, scoring=scoring, fit_params=fit_params,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=False) # return train score is always false, because the scorer gets confused otherwise

    def fit(self, X, y=None, groups=None):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        beamsampler = BeamSampler(self.param_distributions, self.beam_width, self.defaults)
        self.observable_scorer.register(beamsampler)
        res = self._fit(X, y, groups, beamsampler)
        self.defaults = beamsampler.defaults # for unit test
        return res

    def get_params(self, deep=True):
        params = super(BeamSearchCV, self).get_params(deep)
        params['scoring'] = self.original_scoring
        return params
