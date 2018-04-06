import ConfigSpace


def get_neural_network_default_search_space():
    config_space = ConfigSpace.ConfigurationSpace()
    config_space.add_hyperparameter(
        ConfigSpace.CategoricalHyperparameter('imputation:strategy', ['mean', 'median', 'most_frequent']))
    config_space.add_hyperparameter(ConfigSpace.CategoricalHyperparameter('classifier:__choice__', ['neural_network']))
    config_space.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter('classifier:neural_network:hidden_layer_sizes', 32, 1024))
    config_space.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter('classifier:neural_network:num_hidden_layers', 1, 5))
    config_space.add_hyperparameter(
        ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:learning_rate_init', 0.00001, 1, log=True))
    config_space.add_hyperparameter(
        ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:alpha', 0.0000001, 0.0001, log=True))
    # config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:beta_1', 0, 1))
    # config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:beta_2', 0, 1))
    # config_space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter('classifier:neural_network:max_iter', 2, 1000))
    config_space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter('classifier:neural_network:momentum', 0.1, 0.9))
    return config_space
