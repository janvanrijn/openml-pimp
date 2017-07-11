import sklearn
import openmlpimp
import random

from openml.flows import flow_to_sklearn
from openmlstudy14.preprocessing import ConditionalImputer
from sklearn.svm import SVC


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter


def obtain_classifier(configuration_space, indices, max_attempts=5):
    for i in range(max_attempts):
        try:
            configuration = configuration_space.sample_configuration(1)
            classifier = openmlpimp.utils.config_to_classifier(configuration, indices)
            return classifier
        except ValueError:
            # sometimes a classifier is not valid. TODO, check this
            pass


def config_to_classifier(config, indices):
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
        elif param == 'classifier:random_forest:max_features':
            # exception ..
            value = random.uniform(0.1, 0.9)
        else:
            # normal case
            param_name = splitted[0] + '__' + splitted[-1]

        if isinstance(value, str) and value == 'None':
            value = None

        if value == 'True':
            value = True
        elif value == 'False':
            value = False

        pipeline_parameters[param_name] = value

    classifier = None
    if model_type == 'adaboost':
        classifier = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier())
    elif model_type == 'decision_tree':
        classifier = sklearn.tree.DecisionTreeClassifier()
    elif model_type == 'libsvm_svc':
        classifier = SVC()
        pipeline_parameters['classifier__probability'] = True
    elif model_type == 'sgd':
        classifier = sklearn.linear_model.SGDClassifier()
    elif model_type == 'random_forest':
        classifier = sklearn.ensemble.RandomForestClassifier()
    else:
        raise ValueError('Unknown classifier: %s' %classifier)

    steps = [('imputation', ConditionalImputer(strategy='median',
                                               fill_empty=0,
                                               categorical_features=indices,
                                               strategy_nominal='most_frequent')),
             ('hotencoding', sklearn.preprocessing.OneHotEncoder(sparse=False,
                                                                 handle_unknown='ignore',
                                                                 categorical_features=indices)),
             ('variencethreshold', sklearn.feature_selection.VarianceThreshold()),
             ('classifier', classifier)]
    classifier = sklearn.pipeline.Pipeline(steps=steps)
    classifier.set_params(**pipeline_parameters)
    return classifier


def setups_to_configspace(setups, keyfield='parameter_name', ignore_constants=True):
    # setups is result from openml.setups.list_setups call
    # note that this config space is not equal to the one
    # obtained from auto-sklearn; but useful for creating
    # the pcs file
    parameter_values = {}
    flow_id = None
    for setup_id in setups:
        current = setups[setup_id]
        if flow_id is None:
            flow_id = current.flow_id
        else:
            if current.flow_id != flow_id:
                raise ValueError('flow ids are expected to be equal. Expected %d, saw %s' %(flow_id, current.flow_id))

        for param_id in current.parameters.keys():
            name = getattr(current.parameters[param_id], keyfield)
            value = current.parameters[param_id].value
            if name not in parameter_values.keys():
                parameter_values[name] = set()
            parameter_values[name].add(value)

    def is_castable_to(value, type):
        try:
            type(value)
            return True
        except ValueError:
            return False


    cs = ConfigurationSpace()
    constants = set()
    for name in parameter_values.keys():
        all_values = parameter_values[name]
        if len(all_values) <= 1:
            constants.add(name)
            if ignore_constants:
                continue

        if all(is_castable_to(item, int) for item in all_values):
            all_values = [int(item) for item in all_values]
            lower = min(all_values)
            upper = max(all_values)
            hyper = UniformIntegerHyperparameter(name=name,
                                                 lower=lower,
                                                 upper=upper,
                                                 default=int(lower+(upper-lower) / 2),  # TODO don't know
                                                 log=False)                             # TODO don't know
            cs.add_hyperparameter(hyper)
        elif all(is_castable_to(item, float) for item in all_values):
            all_values = [float(item) for item in all_values]
            lower = min(all_values)
            upper = max(all_values)
            hyper = UniformFloatHyperparameter(name=name,
                                               lower=lower,
                                               upper=upper,
                                               default=lower + (upper - lower) / 2,  # TODO don't know
                                               log=False)                            # TODO don't know
            cs.add_hyperparameter(hyper)
        else:
            values = [flow_to_sklearn(item) for item in all_values]
            hyper = CategoricalHyperparameter(name=name,
                                              choices=values,
                                              default=values[0]) # TODO don't know
            cs.add_hyperparameter(hyper)
    return cs, constants


def runhistory_to_trajectory(runhistory, default_setup_id):
    trajectory_lines = []
    lowest_cost = 1.0
    lowest_cost_idx = None
    default_cost = None

    for run in runhistory['data']:
        config_id = run[0][0] # magic index
        cost = run[1][0] # magic index
        print(cost)
        if cost < lowest_cost:
            lowest_cost = cost
            lowest_cost_idx = config_id

        if config_id == default_setup_id:
            if default_cost is not None:
                raise ValueError('default setup id should be encountered once')
            default_cost = run[1][0] # magic index

    if default_cost is None:
        raise ValueError('could not find default setup')

    if default_cost == lowest_cost:
        raise ValueError('no improvement over default param settings')

    if lowest_cost_idx == default_setup_id:
        raise ValueError('default setup id should not be best performing algorithm')

    def _default_trajectory_line():
        return {"cpu_time": 0.0, "evaluations": 0, "total_cpu_time": 0.0, "wallclock_time": 0.0 }

    def paramdict_to_incumbent(param_dict):
        res = []
        for param in param_dict.keys():
            res.append(param + "='" + str(param_dict[param]) + "'")
        return res

    initial = _default_trajectory_line()
    initial['cost'] = default_cost
    initial['incumbent'] = paramdict_to_incumbent(runhistory['configs'][default_setup_id])
    trajectory_lines.append(initial)

    final = _default_trajectory_line()
    final['cost'] = lowest_cost
    final['incumbent'] = paramdict_to_incumbent(runhistory['configs'][lowest_cost_idx])
    trajectory_lines.append(final)

    return trajectory_lines
