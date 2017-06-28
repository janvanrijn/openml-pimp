import sklearn

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

def config_to_decision_tree(config):
    parameter_settings = config.get_dictionary()
    if parameter_settings['classifier:__choice__'] != 'decision_tree':
        raise ValueError('Can only instantiate Decision Trees')

    classifier = sklearn.pipeline.Pipeline(steps=[('imputation', sklearn.preprocessing.Imputer()),
                                                  ('classifier', sklearn.tree.DecisionTreeClassifier())])

    pipeline_parameters = {}
    for param, value in parameter_settings.items():

        splitted = param.split(':')
        if splitted[0] not in ['imputation', 'classifier']:
            continue
        if splitted[1] == '__choice__':
            continue
        param_name = splitted[0] + '__' + splitted[-1]

        # TODO: hack
        if isinstance(value, str) and value == 'None':
            value = None

        pipeline_parameters[param_name] = value
    classifier.set_params(**pipeline_parameters)
    return classifier


def setups_to_configspace(setups):
    # note that this config space is not equal to the one
    # obtained from auto-sklearn; but useful for creating
    # the pcs file
    parameter_values = {}
    flow_id = None
    for setup_id in setups:
        current = setups[setup_id]
        if flow_id is not None:
            flow_id = current.flow_id
        else:
            if current.flow_id != flow_id:
                raise ValueError('flow ids are expected to be equal')

        name = current.parameter_name
        value = current.value
        if name not in parameter_values.keys():
            parameter_values[name] = set()
        parameter_values[name].add(value)

    cs = ConfigurationSpace()
    for name in parameter_values.keys():
        all_values = parameter_values[name]
        lower = min(all_values)
        upper = max(all_values)
        if len(all_values) <= 1:
            # constant
            continue
        elif all(isinstance(item, int) for item in all_values):
            hyper = UniformIntegerHyperparameter(name=name,
                                                 lower=lower,
                                                 upper=upper,
                                                 default=lower+(upper-lower)/2, # TODO don't know
                                                 log=False)                     # TODO don't know
            cs.add_hyperparameter(hyper)
        elif all(isinstance(item, float) for item in all_values):
            hyper = UniformFloatHyperparameter(name=name,
                                                 lower=lower,
                                                 upper=upper,
                                                 default=lower + (upper - lower) / 2,  # TODO don't know
                                                 log=False)                            # TODO don't know
            cs.add_hyperparameter(hyper)
        else:
            values = list(all_values)
            hyper = CategoricalHyperparameter(name=name,
                                              choices=values,
                                              default=values[0]) # TODO don't know
            cs.add_hyperparameter(hyper)
    return cs
