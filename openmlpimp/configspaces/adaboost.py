from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
UnParametrizedHyperparameter



def get_adaboost_default_search_space(dataset_properties=None):
    classif_prefix = 'classifier:adaboost:'

    cs = ConfigurationSpace()

    model_type = CategoricalHyperparameter('classifier:__choice__', ['adaboost'])
    imputation = CategoricalHyperparameter('imputation:strategy', ['mean', 'median', 'most_frequent'])

    n_estimators = UniformIntegerHyperparameter(
        name=classif_prefix + "n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = UniformFloatHyperparameter(
        name=classif_prefix + "learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = CategoricalHyperparameter(
        name=classif_prefix + "algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
    max_depth = UniformIntegerHyperparameter(
        name=classif_prefix + "max_depth", lower=1, upper=10, default_value=1, log=False)

    cs.add_hyperparameters([model_type, imputation, n_estimators, learning_rate, algorithm, max_depth])

    return cs
