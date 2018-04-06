import ConfigSpace
import openmlpimp


def get_config_space(classifier, type='default'):
    fn_string = 'get_' + type + '_' + classifier + '_search_space'
    fn_reference = getattr(openmlpimp.configspaces, fn_string)
    cf_space_raw = fn_reference()

    # post processing # TODO: integrate
    configuration_space = ConfigSpace.ConfigurationSpace()
    for name, hyperparameter in cf_space_raw._hyperparameters.items():
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
