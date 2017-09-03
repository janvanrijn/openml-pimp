import ConfigSpace
import autosklearn.constants
from autosklearn.util.pipeline import get_configuration_space


def get_config_space(classifier):
    if classifier is not 'neural_network':
        autosklearn_config_space = get_configuration_space(
            info={'task': autosklearn.constants.MULTICLASS_CLASSIFICATION, 'is_sparse': 0},
            include_estimators=[classifier],
            include_preprocessors=['no_preprocessing'])

        configuration_space = ConfigSpace.ConfigurationSpace()
        for name, hyperparameter in autosklearn_config_space._hyperparameters.items():
            if isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant):
                continue
            if hyperparameter.name.startswith('classifier') or hyperparameter.name.startswith('imputation'):
                configuration_space.add_hyperparameter(hyperparameter)

        if classifier == 'random_forest':
            hyperparameter = configuration_space.get_hyperparameter('classifier:random_forest:max_features')
            hyperparameter.lower = 0.1
            hyperparameter.lower_hard = 0.1
            hyperparameter.upper = 0.9
            hyperparameter.upper_hard = 0.9
            hyperparameter.default = 0.1

        return configuration_space

    config_space = ConfigSpace.ConfigurationSpace()
    config_space.add_hyperparameter(ConfigSpace.CategoricalHyperparameter('imputation:strategy', ['mean', 'median', 'most_frequent']))
    config_space.add_hyperparameter(ConfigSpace.CategoricalHyperparameter('classifier:__choice__', [classifier]))
    config_space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter('classifier:neural_network:hidden_layer_sizes', 32, 1024))
    config_space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter('classifier:neural_network:num_hidden_layers', 1, 5))
    config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:learning_rate_init', 0.00001, 1, log=True))
    config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:alpha', 0.0000001, 0.0001, log=True))
   # config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:beta_1', 0, 1))
   # config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:beta_2', 0, 1))
   # config_space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter('classifier:neural_network:max_iter', 2, 1000))
    config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:momentum', 0.1, 0.9))
    return config_space


def get_config_space_casualnames(classifier, fixed_parameters=None):
    config_space = get_config_space(classifier)
    config_space_prime = ConfigSpace.ConfigurationSpace()
    for name, hyperparameter in config_space._hyperparameters.items():
        if name == 'classifier:__choice__':
            continue
        casualname = hyperparameter.name.split(':')[-1]
        if fixed_parameters is not None and casualname in fixed_parameters:
            continue

        if isinstance(hyperparameter, ConfigSpace.CategoricalHyperparameter):
            config_space_prime.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(casualname, hyperparameter.choices, default=hyperparameter.default))
        elif isinstance(hyperparameter, ConfigSpace.UniformIntegerHyperparameter):
            config_space_prime.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(casualname, hyperparameter.lower, hyperparameter.upper, log=hyperparameter.log, default=hyperparameter.default))
        elif isinstance(hyperparameter, ConfigSpace.UniformFloatHyperparameter):
            config_space_prime.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(casualname, hyperparameter.lower, hyperparameter.upper, log=hyperparameter.log, default=hyperparameter.default))
        else:
            raise ValueError()
    return config_space_prime
