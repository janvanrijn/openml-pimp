from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    UnParametrizedHyperparameter, OrdinalHyperparameter


def get_libsvm_svc_default_search_space():
    classif_prefix = "classifier:libsvm_svc:"

    model_type = CategoricalHyperparameter('classifier:__choice__', ['libsvm_svc'])
    imputation = CategoricalHyperparameter('imputation:strategy', ['mean', 'median', 'most_frequent'])

    C = UniformFloatHyperparameter(classif_prefix + "C", 0.03125, 32768, log=True, default_value=1.0)
    # No linear kernel here, because we have liblinear
    kernel = CategoricalHyperparameter(name=classif_prefix + "kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf")
    degree = UniformIntegerHyperparameter(classif_prefix + "degree", 2, 5, default_value=3)
    gamma = UniformFloatHyperparameter(classif_prefix + "gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
    # TODO this is totally ad-hoc
    coef0 = UniformFloatHyperparameter(classif_prefix + "coef0", -1, 1, default_value=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = CategoricalHyperparameter(classif_prefix + "shrinking", ["True", "False"], default_value="True")
    tol = UniformFloatHyperparameter(classif_prefix + "tol", 1e-5, 1e-1, default_value=1e-3, log=True)
    # cache size is not a hyperparameter, but an argument to the program!
    max_iter = UnParametrizedHyperparameter(classif_prefix + "max_iter", -1)

    cs = ConfigurationSpace()
    cs.add_hyperparameters([model_type, imputation, C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])

    degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
    coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs


def get_libsvm_svc_extended_search_space():
    classif_prefix = "classifier:libsvm_svc:"

    model_type = CategoricalHyperparameter('classifier:__choice__', ['libsvm_svc'])
    imputation = CategoricalHyperparameter('imputation:strategy', ['mean', 'median', 'most_frequent'])

    C = UniformFloatHyperparameter(classif_prefix + "C", 0.03125, 32768 * 4, log=True, default_value=1.0)
    # No linear kernel here, because we have liblinear
    kernel = CategoricalHyperparameter(name=classif_prefix + "kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf")
    degree = UniformIntegerHyperparameter(classif_prefix + "degree", 2, 5, default_value=3)
    gamma = UniformFloatHyperparameter(classif_prefix + "gamma", 3.0517578125e-05 / 4, 8, log=True, default_value=0.1)
    # TODO this is totally ad-hoc
    coef0 = UniformFloatHyperparameter(classif_prefix + "coef0", -1, 1, default_value=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = CategoricalHyperparameter(classif_prefix + "shrinking", ["True", "False"], default_value="True")
    tol = UniformFloatHyperparameter(classif_prefix + "tol", 1e-5, 1e-1, default_value=1e-3, log=True)
    # cache size is not a hyperparameter, but an argument to the program!
    max_iter = UnParametrizedHyperparameter(classif_prefix + "max_iter", -1)

    cs = ConfigurationSpace()
    cs.add_hyperparameters([model_type, imputation, C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])

    degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
    coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs
