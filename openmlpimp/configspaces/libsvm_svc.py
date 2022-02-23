from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, UnParametrizedHyperparameter


def get_hyperparameter_search_space(seed):

    imputation = CategoricalHyperparameter('columntransformer__numeric__imputer__strategy', ['mean', 'median', 'most_frequent'])

    C = UniformFloatHyperparameter("svc__C", 0.03125, 32768, log=True, default_value=1.0)
    # No linear kernel here, because we have liblinear
    kernel = CategoricalHyperparameter(name="svc__kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf")
    #degree = UniformIntegerHyperparameter("svc__degree", 1, 5, default_value=3)
    gamma = UniformFloatHyperparameter("svc__gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
    # TODO this is totally ad-hoc
    #coef0 = UniformFloatHyperparameter("svc__coef0", -1, 1, default_value=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = CategoricalHyperparameter("svc__shrinking", [True, False], default_value=True)
    tol = UniformFloatHyperparameter("svc__tol", 1e-5, 1e-1, default_value=1e-3, log=True)
    # cache size is not a hyperparameter, but an argument to the program!
    # max_iter = UnParametrizedHyperparameter("svc__max_iter", -1)

    cs = ConfigurationSpace('sklearn.svm.SVC', seed)
    cs.add_hyperparameters([imputation, C, kernel, gamma, shrinking, tol])


    return cs

