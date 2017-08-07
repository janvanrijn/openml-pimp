import os
import json

import numpy as np

from matplotlib import pyplot as plt

from ConfigSpace.hyperparameters import CategoricalHyperparameter

from fanova.fanova import fANOVA as fanova_pyrfr
from fanova.visualizer import Visualizer


class FanovaBackend(object):

    @staticmethod
    def _plot_result(fANOVA, configspace, directory):
        vis = Visualizer(fANOVA, configspace)

        try: os.makedirs(directory)
        except FileExistsError: pass

        for hp in configspace.get_hyperparameters():
            plt.close('all')
            plt.clf()
            param = hp.name
            outfile_name = os.path.join(directory, param.replace(os.sep, "_") + ".png")
            if isinstance(hp, (CategoricalHyperparameter)):
                vis.plot_categorical_marginal(configspace.get_idx_by_hyperparameter_name(param), show=False)
            else:
                vis.plot_marginal(configspace.get_idx_by_hyperparameter_name(param), show=False)
            plt.savefig(outfile_name)
        pass


    @staticmethod
    def execute(save_folder, runhistory, configspace):
        try: os.makedirs(save_folder)
        except FileExistsError: pass

        X = []
        y = []

        for item in runhistory['data']:
            current = []
            setup_id = str(item[0][0])
            configuration = runhistory['configs'][setup_id]
            for param in configspace.get_hyperparameters():
                value = configuration[param.name]
                if isinstance(param, CategoricalHyperparameter):
                    value = param.choices.index(value)
                current.append(value)
            X.append(current)
            y.append(item[1][0])
        X = np.array(X)
        y = np.array(y)

        max_tries = 5
        for i in range(max_tries):
            try:
                # start the evaluator
                evaluator = fanova_pyrfr(X=X, Y=y, config_space=configspace, config_on_hypercube=False)
                # obtain the results
                params = configspace.get_hyperparameters()
                result = {}
                for idx, param in enumerate(params):
                    importance = evaluator.quantify_importance([idx])[(idx,)]['total importance']
                    result[param.name] = importance
                # store to disk
                filename = 'pimp_values_fanova.json'
                with open(os.path.join(save_folder, filename), 'w') as out_file:
                    json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
                # call plotting fn
                FanovaBackend._plot_result(evaluator, configspace, save_folder + "/fanova")
                return save_folder + "/" + filename
            except ZeroDivisionError as e:
                if i + 1 == max_tries:
                    raise e
        raise ValueError('Should never happen. ')