import collections
import openmlpimp

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from scipy.stats import gaussian_kde



from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._search import ParameterSampler

from collections import Sized, defaultdict
from functools import partial

import numpy as np

from sklearn.utils import resample
from sklearn.base import is_classifier, clone
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.fixes import rankdata
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import indexable
from sklearn.metrics.scorer import check_scoring



class BaseSearchBandits(BaseSearchCV):

    def _fit(self, X, y, groups, parameter_iterable, eta, successive_halving_steps):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        results = dict()

        for sample_idx in range(successive_halving_steps - 1, -1, -1):
            sample_size = int(len(X) / eta ** sample_idx)

            if groups is not None:
                X_resampled, y_resampled, groups_resampled = resample(X, y, groups, n_samples=sample_size, replace=False, random_state=self.random_state)
            else:
                X_resampled, y_resampled = resample(X, y, n_samples=sample_size, replace=False)
                groups_resampled = None

            cv_iter = list(cv.split(X_resampled, y_resampled, groups_resampled))
            out = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch
            )(delayed(_fit_and_score)(clone(base_estimator), X_resampled, y_resampled, self.scorer_,
                                      train, test, self.verbose, parameters,
                                      fit_params=self.fit_params,
                                      return_train_score=self.return_train_score,
                                      return_n_test_samples=True,
                                      return_times=True, return_parameters=True,
                                      error_score=self.error_score)
              for parameters in parameter_iterable
              for train, test in cv_iter)

            # if one choose to see train score, "out" will contain train score info
            if self.return_train_score:
                (train_scores, test_scores, test_sample_counts,
                 fit_time, score_time, parameters) = zip(*out)
            else:
                (test_scores, test_sample_counts,
                 fit_time, score_time, parameters) = zip(*out)

            candidate_params = parameters[::n_splits]
            n_candidates = len(candidate_params)
            sample_sizes = [sample_size] * n_candidates * n_splits

            def _store(key_name, array, weights=None, splits=False, rank=False):
                """A small helper to store the scores/times to the cv_results_"""
                array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                                  n_splits)
                if splits:
                    for split_i in range(n_splits):
                        splits_key = "split%d_%s" % (split_i, key_name)
                        if splits_key not in results:
                            results[splits_key] = array[:, split_i]
                        else:
                            results[splits_key] = np.append(results[splits_key], array[:, split_i])

                array_means = np.average(array, axis=1, weights=weights)
                means_key = 'mean_%s' % key_name
                if means_key not in results:
                    results[means_key] = array_means
                else:
                    results[means_key] = np.append(results[means_key], array_means)

                # Weighted std is not directly available in numpy
                array_stds = np.sqrt(np.average((array -
                                                 array_means[:, np.newaxis]) ** 2,
                                                axis=1, weights=weights))
                stds_key = 'std_%s' % key_name
                if stds_key not in results:
                    results[stds_key] = array_stds
                else:
                    results[stds_key] = np.append(results[stds_key], array_stds)

                if rank:
                    ranks_key = "rank_%s" % key_name
                    array_ranks = np.asarray(rankdata(-array_means, method='min'), dtype=np.int32)
                    if ranks_key not in results:
                        results[ranks_key] = array_ranks
                    else:
                        results[ranks_key] = np.append(results[ranks_key], array_ranks)

            # Computed the (weighted) mean and std for test scores alone
            # NOTE test_sample counts (weights) remain the same for all candidates
            test_sample_counts = np.array(test_sample_counts[:n_splits],
                                          dtype=np.int)

            _store('test_score', test_scores, splits=True, rank=True,
                   weights=test_sample_counts if self.iid else None)
            if self.return_train_score:
                _store('train_score', train_scores, splits=True)
            _store('fit_time', fit_time)
            _store('score_time', score_time)
            _store('sample_sizes', sample_sizes)

            best_index = np.flatnonzero(results["rank_test_score"][-n_candidates:] == 1)[0]
            best_parameters = candidate_params[best_index]

            # prepare parameter iterable for next round
            parameter_iterable = []
            order = np.argsort(results['mean_test_score'][-n_candidates:] * -1)
            for i in range(int(n_candidates / eta)):
                parameter_iterable.append(candidate_params[order[i]])

            # Use one MaskedArray and mask all the places where the param is not
            # applicable for that candidate. Use defaultdict as each candidate may
            # not contain all the params
            param_results = defaultdict(partial(MaskedArray,
                                                np.empty(n_candidates,),
                                                mask=True,
                                                dtype=object))
            for cand_i, params in enumerate(candidate_params):
                for name, value in params.items():
                    # An all masked empty array gets created for the key
                    # `"param_%s" % name` at the first occurence of `name`.
                    # Setting the value at an index also unmasks that index
                    param_key_name = "param_%s" % name
                    param_results[param_key_name][cand_i] = value

            for param in param_results.keys():
                if param not in results:
                    results[param] = param_results[param]
                else:
                    results[param] = np.append(results[param], param_results[param])

            # Store a list of param dicts at the key 'params'
            if 'params' not in results:
                results['params'] = candidate_params
            else:
                results['params'] = results['params'] + candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


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



class SuccessiveHalving(BaseSearchBandits):

    def __init__(self, estimator, param_distributions, successive_halving_steps,
                 eta, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=True):
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.successive_halving_steps = successive_halving_steps
        self.eta = eta
        super(SuccessiveHalving, self).__init__(
             estimator=estimator, scoring=scoring, fit_params=fit_params,
             n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)

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
        num_arms = self.eta ** (self.successive_halving_steps - 1)
        sampled_params = ParameterSampler(self.param_distributions,
                                          num_arms,
                                          random_state=self.random_state)
        return self._fit(X, y, groups, sampled_params, self.eta, self.successive_halving_steps)
