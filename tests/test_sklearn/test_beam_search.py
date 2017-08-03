
import unittest

from collections import OrderedDict

from sklearn.model_selection._search import RandomizedSearchCV
from openmlpimp.sklearn.beam_search import BeamSearchCV
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

class SklearnWrapperTest(unittest.TestCase):

    def test_beam_search(self):
        iris = datasets.load_iris()

        param_grid = OrderedDict()
        param_grid['max_leaf_nodes'] = [4, 5, 6, 7, 8, 9, 10]
        param_grid['min_samples_leaf'] = [1, 2, 4, 8, 16, 32, 64, 128]
        param_grid['max_depth'] = [2, 4, 8, 16, 32]
        param_grid['criterion'] = ['gini', 'entropy']

        tree = DecisionTreeClassifier()
        default_params = tree.get_params()

        model = BeamSearchCV(
            estimator=tree,
            param_distributions=param_grid)
        model.fit(iris.data, iris.target)
        results = model.predict(iris.data)

        self.assertEquals(len(results), len(iris.target))

        self.assertEqual(max(model.cv_results_['mean_test_score']), model.best_score_)

        # assert default params before optimization, optimized value after optimization
        current_param_idx = 0
        last_update = 0
        for cv_idx in range(len(model.cv_results_)):
            if cv_idx - last_update == len(param_grid[list(param_grid.keys())[current_param_idx]]):
                current_param_idx += 1
                last_update = cv_idx

            for param_idx, param in enumerate(param_grid.keys()):
                if param_idx < current_param_idx:
                    self.assertEquals(model.cv_results_['param_'+param][cv_idx], model.best_params_[param])
                elif param_idx > current_param_idx:
                    self.assertEquals(model.cv_results_['param_'+param][cv_idx], default_params[param])
