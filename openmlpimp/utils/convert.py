import sklearn

from sklearn.svm import SVC

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter


def config_to_classifier(config):
    parameter_settings = config.get_dictionary()

    model_type = None
    pipeline_parameters = {}
    for param, value in parameter_settings.items():

        splitted = param.split(':')
        if splitted[0] not in ['imputation', 'classifier']:
            continue
        elif splitted[1] == '__choice__':
            if splitted[0] == 'classifier':
                model_type = value
            continue
        elif param == 'classifier:adaboost:max_depth':
            # exception ..
            param_name = 'classifier__base_estimator__max_depth'
        else:
            # normal case
            param_name = splitted[0] + '__' + splitted[-1]

        # TODO: hack
        if isinstance(value, str) and value == 'None':
            value = None

        pipeline_parameters[param_name] = value

    classifier = None
    if model_type == 'adaboost':
        classifier = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier())
    elif model_type == 'decision_tree':
        classifier = sklearn.tree.DecisionTreeClassifier()
    elif model_type == 'libsvm_svc':
        classifier = SVC()
    elif model_type == 'sgd':
        classifier = sklearn.linear_model.SGDClassifier()
    elif model_type == 'random_forest':
        classifier = sklearn.ensemble.RandomForestClassifier()
    else:
        raise ValueError('Unknown classifier: %s' %classifier)


    classifier = sklearn.pipeline.Pipeline(steps=[('imputation', sklearn.preprocessing.Imputer()),
                                                  ('classifier', classifier)])
    print(pipeline_parameters)
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
