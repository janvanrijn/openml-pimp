import copy

import numpy as np

from sklearn.model_selection import BaseSearchCV
from sklearn.model_selection import ParameterSampler


class SuccessiveHalving(BaseSearchCV):

    def __init__(self, estimator, param_distributions, budget_parameter,
                 budget_max, num_steps=4, eta=3, scoring=None,
                 fit_params=None, n_jobs=1, iid='warn', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise-deprecating', return_train_score="warn"):
        self.param_distributions = param_distributions
        self.budget_parameter = budget_parameter
        self.budget_max = budget_max
        self.num_steps = num_steps
        self.eta = eta
        self.random_state = random_state

        super(SuccessiveHalving, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _generate_candidates(self):
        """Return ParameterSampler instance for the given distributions"""

        # Hyperband parameter. Budget per elementary run. in the paper, this is multiplied by R
        hyperband_B = self.num_steps + 1
        hyperband_r = self.budget_max * self.eta ** (-1 * (self.num_steps-1))

        # Hyperband parameter. initial number of configurations
        hyperband_n = int(np.ceil(hyperband_B * self.eta ** self.num_steps / (self.num_steps + 1)))

        # spoof sample distribution to have the budget parameter
        budgetted_distributions = copy.deepcopy(self.param_distributions)
        budgetted_distributions[self.budget_parameter] = [int(hyperband_r * self.eta ** 0)]

        # in the first iteration we actually sample settings
        result = yield ParameterSampler(budgetted_distributions, hyperband_n, random_state=self.random_state)

        for sh_iteration in range(1, self.num_steps + 1):  # loop: for i in {0, ..., s} do
            # select top k
            remaining = int(np.floor(len(result['mean_test_score']) / self.eta))
            order = np.argsort(result['mean_test_score'])
            best_params = np.array(result['params'])[order[:remaining]]

            # add budget parameter
            for i in range(len(best_params)):
                best_params[i][self.budget_parameter] = int(hyperband_r * self.eta ** sh_iteration)

            result = yield best_params
            # remove 'old' parameter settings (from previous iterations)
            result = {key: value[-len(best_params):] for key, value in result.items()}
