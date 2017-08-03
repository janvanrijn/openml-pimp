import numpy as np
import copy

from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics.scorer import check_scoring


class BeamSampler(object):
    """In accordance with the Observer pattern """

    def __init__(self, param_distributions, beam_width, defaults):
        self.param_distributions = param_distributions
        self.beam_width = beam_width
        self.recent_results = dict()
        self.param_order = []
        self.defaults = defaults
        if set(defaults.keys()) != set(self.param_distributions.keys()):
            raise ValueError('Param distributions and defaults do not agree. ')

        all_lists = np.all([not hasattr(v, "rvs") for v in self.param_distributions.values()])
        if not all_lists:
            raise ValueError('Currently only lists of parameter values are supported. ')
        if self.beam_width == 1:
            self.n_iter = sum([len(param_distributions[key]) for key in param_distributions])
        else:
            raise ValueError('Currently, only beam_width = 1 is supported. ')

    def __iter__(self):
        params = self.defaults
        for param, values in self.param_distributions.items():
            self.param_order.append(param)
            for value in values:
                params[param] = value
                yield copy.deepcopy(params)
            # now obtain the value of params that performed best and fix it
            params[param] = max(self.recent_results, key=lambda i: np.mean(self.recent_results[i]))
            print(param, params[param])
            for val, res in self.recent_results.items(): print(val, np.mean(res))
            # flush the memory of collected results
            self.recent_results = dict()

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter

    def update(self, params, score):
        if len(self.param_order) == 0:
            raise ValueError('param_order is None, call to update function unexpected. ')
        current_value = params[self.param_order[-1]]
        if current_value not in self.recent_results:
            self.recent_results[current_value] = []
        self.recent_results[current_value].append(score)


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
        self.defaults = {}
        for param in param_distributions.keys():
            self.defaults[param] = estimator.get_params()[param]

        # trick to communicate scores back to param grid.
        original_scoring = check_scoring(estimator, scoring)
        self.observable_scorer = ObservableScorer(original_scoring)
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
        return self._fit(X, y, groups, beamsampler)
