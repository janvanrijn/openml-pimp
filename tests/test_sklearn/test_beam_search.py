
import unittest

from collections import OrderedDict

from openmlpimp.sklearn.beam_search import BeamSearchCV
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


class SklearnWrapperTest(unittest.TestCase):

    def _execute_test(self, classifier, param_grid, dataset):
        param_order = list(param_grid.keys())

        model = BeamSearchCV(
            estimator=classifier,
            param_distributions=param_grid)
        model.fit(dataset.data, dataset.target)
        default_values = model.defaults

        results = model.predict(dataset.data)

        self.assertEquals(len(results), len(dataset.target))

        self.assertEqual(len(model.cv_results_['mean_test_score']), 1 + sum([len(values)-1 for param, values in param_grid.items()]))

        self.assertEqual(max(model.cv_results_['mean_test_score']), model.best_score_)

        # assert default params before optimization, optimized value after optimization
        current_param_idx = 0
        last_update = 0
        for cv_idx in range(len(model.cv_results_['mean_test_score'])):
            current_param = param_order[current_param_idx]
            num_values = len(param_grid[current_param])
            if current_param_idx > 0:
                num_values -= 1

            if cv_idx - last_update == num_values:
                current_param_idx += 1
                last_update = cv_idx
                current_param = param_order[current_param_idx]

            # print(cv_idx, ":", current_param_idx, current_param)
            for param_idx, param in enumerate(param_grid.keys()):
                # print(param_idx, param, default_values[param], model.cv_results_['param_'+param][cv_idx])
                if param_idx < current_param_idx:
                    self.assertEquals(model.cv_results_['param_'+param][cv_idx], model.best_params_[param])
                elif param_idx > current_param_idx:
                    self.assertEquals(model.cv_results_['param_'+param][cv_idx], default_values[param])

    def test_beam_search(self):
        iris = datasets.load_iris()

        param_grid = OrderedDict()
        param_grid['max_leaf_nodes'] = [None, 4, 5, 6, 7, 8, 9, 10]
        param_grid['min_samples_leaf'] = [1, 2, 4, 8, 16, 32, 64, 128]
        param_grid['max_depth'] = [None, 2, 4]
        param_grid['criterion'] = ['gini', 'entropy']

        tree = DecisionTreeClassifier()
        self._execute_test(tree, param_grid, iris)

    def test_beam_search_missingdefault(self):
        iris = datasets.load_iris()

        param_grid = OrderedDict()
        param_grid['max_leaf_nodes'] = [4, 5, 6, 7, 8, 9, 10]
        param_grid['min_samples_leaf'] = [2, 8, 32, 128]
        param_grid['max_depth'] = [2, 4]
        param_grid['criterion'] = ['gini', 'entropy']

        tree = DecisionTreeClassifier()
        self._execute_test(tree, param_grid, iris)
