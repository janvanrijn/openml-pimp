from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
UnParametrizedHyperparameter, Constant


def get_random_forest_default_search_space():
    classif_prefix = 'classifier:random_forest:'

    cs = ConfigurationSpace()
    model_type = CategoricalHyperparameter('classifier:__choice__', ['random_forest'])
    imputation = CategoricalHyperparameter('imputation:strategy', ['mean', 'median', 'most_frequent'])
    n_estimators = Constant(classif_prefix + "n_estimators", 100)
    criterion = CategoricalHyperparameter(
        classif_prefix + "criterion", ["gini", "entropy"], default_value="gini")

    # The maximum number of features used in the forest is calculated as m^max_features, where
    # m is the total number of features, and max_features is the hyperparameter specified below.
    # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
    # corresponds with Geurts' heuristic.
    max_features = UniformFloatHyperparameter(
        classif_prefix + "max_features", 0., 1., default_value=0.5)

    max_depth = UnParametrizedHyperparameter(classif_prefix + "max_depth", "None")
    min_samples_split = UniformIntegerHyperparameter(
        classif_prefix + "min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        classif_prefix + "min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = UnParametrizedHyperparameter(classif_prefix + "min_weight_fraction_leaf", 0.)
    max_leaf_nodes = UnParametrizedHyperparameter(classif_prefix + "max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter(classif_prefix + 'min_impurity_decrease', 0.0)
    bootstrap = CategoricalHyperparameter(
        classif_prefix + "bootstrap", ["True", "False"], default_value="True")
    cs.add_hyperparameters([model_type, imputation, n_estimators, criterion, max_features,
                            max_depth, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            bootstrap, min_impurity_decrease])


    return cs
