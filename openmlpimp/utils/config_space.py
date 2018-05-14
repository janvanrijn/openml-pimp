import ConfigSpace
import json
import openmlpimp


def openmlsetup_to_configuration(openmlsetup, config_space):
    name_values = dict()
    for param_id, param in openmlsetup.parameters.items():
        name = param.parameter_name
        if name in config_space.get_hyperparameter_names():
            hyperparam = config_space._hyperparameters[name]
            if isinstance(hyperparam, ConfigSpace.hyperparameters.UniformIntegerHyperparameter):
                name_values[name] = int(param.value)
            elif isinstance(hyperparam, ConfigSpace.hyperparameters.NumericalHyperparameter):
                name_values[name] = float(param.value)
            else:
                val = json.loads(param.value)
                if isinstance(val, bool):
                    val = str(val)
                name_values[name] = val

    return ConfigSpace.Configuration(config_space, name_values)


def get_config_space(classifier, type='default'):
    fn_string = 'get_' + classifier + '_' + type + '_search_space'
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
        # hyperparameter.lower_hard = 0.1
        hyperparameter.upper = 0.9
        # hyperparameter.upper_hard = 0.9
        hyperparameter.default_value = 0.1

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
            config_space_prime.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(casualname, hyperparameter.choices, default_value=hyperparameter.default_value))
        elif isinstance(hyperparameter, ConfigSpace.UniformIntegerHyperparameter):
            config_space_prime.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(casualname, hyperparameter.lower, hyperparameter.upper, log=hyperparameter.log, default_value=hyperparameter.default_value))
        elif isinstance(hyperparameter, ConfigSpace.UniformFloatHyperparameter):
            config_space_prime.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(casualname, hyperparameter.lower, hyperparameter.upper, log=hyperparameter.log, default_value=hyperparameter.default_value))
        else:
            raise ValueError()
    return config_space_prime
