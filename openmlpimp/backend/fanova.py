import os
import json
import openmlpimp

import numpy as np

from matplotlib import pyplot as plt

from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.io.pcs_new import read

from fanova.fanova import fANOVA as fanova_pyrfr
from fanova.visualizer import Visualizer


class FanovaBackend(object):

    @staticmethod
    def _plot_result(fANOVA, configspace, directory, yrange=None):
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
                vis.plot_marginal(configspace.get_idx_by_hyperparameter_name(param), resolution=100, show=False)
            x1, x2, _, _ = plt.axis()
            if yrange:
                plt.axis((x1, x2, yrange[0], yrange[1]))
            plt.savefig(outfile_name)
        pass


    @staticmethod
    def execute(save_folder, runhistory_location, configspace_location, manual_logtransform=True, use_percentiles=True, interaction_effect=True):
        with open(runhistory_location) as runhistory_file:
            runhistory = json.load(runhistory_file)
        with open(configspace_location) as configspace_file:
            configspace = read(configspace_file)

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
                elif param.log and manual_logtransform:
                    value = np.log(value)

                current.append(value)
            X.append(current)
            y.append(item[1][0])
        X = np.array(X)
        y = np.array(y)

        if manual_logtransform:
            configspace = openmlpimp.utils.scale_configspace_to_log(configspace)

        if use_percentiles:
            p75 = np.percentile(y, 75.0)
            p100 = np.percentile(y, 100.0)

        # start the evaluator
        evaluator = fanova_pyrfr(X=X, Y=y, config_space=configspace, config_on_hypercube=False, cutoffs=(p75, p100))
        # obtain the results
        params = configspace.get_hyperparameters()
        result = {}

        for idx, param in enumerate(params):
            importance = evaluator.quantify_importance([idx])[(idx,)]['total importance']
            result[param.name] = importance

        # store main results to disk
        filename = 'pimp_values_fanova.json'
        with open(os.path.join(save_folder, filename), 'w') as out_file:
            json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))

        # call plotting fn
        yrange = (0, 1)
        if use_percentiles:
            yrange = (p75, p100)
        FanovaBackend._plot_result(evaluator, configspace, save_folder + '/fanova', yrange)

        # if interaction_effect:
        #     result_interaction = {}
        #     for idx, param in enumerate(params):
        #         for idx2, param2 in enumerate(params):
        #             if idx2 == idx:
        #                 continue
        #             interaction = evaluator.quantify_importance([idx, idx2])[(idx,idx2)]['total importance']
        #             interaction -= result[param.name]
        #             interaction -= result[param2.name]
        #             if interaction < 0.0:
        #                 raise ValueError()
        #             result_interaction[param.name + '__' + param2.name] = interaction
        #             result_interaction[param2.name + '__' + param.name] = interaction
        #
        #     # store interaction effects to disk
        #     with open(os.path.join(save_folder, 'pimp_values_fanova_interaction.json'), 'w') as out_file:
        #         json.dump(result_interaction, out_file, sort_keys=True, indent=4, separators=(',', ': '))
        #     # vis = Visualizer(evaluator, configspace)
        #     # vis.create_most_important_pairwise_marginal_plots(save_folder + '/fanova')

        return save_folder + "/" + filename
