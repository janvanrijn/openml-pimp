import openmlpimp
import sklearn.datasets
import sklearn.ensemble
import unittest


class VerifySuccessiveHalvingRunTest(unittest.TestCase):

    def test_successive_halving(self):
        param_dist = {
            'min_samples_leaf': range(1, 20),
            'min_samples_split': range(2, 20),
            'max_depth': range(1, 10)
        }

        rf = sklearn.ensemble.RandomForestClassifier()
        X, y = sklearn.datasets.load_digits(return_X_y=True)

        sh = openmlpimp.search.SuccessiveHalving(rf, param_dist, 'n_estimators', 8, num_steps=4, eta=2)
        sh.fit(X, y)

        evaluated_params = sh.cv_results_['params']

        # TODO: test

