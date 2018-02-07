
import unittest

import arff
import math
import numpy as np
import openmlpimp
import os


class VerifySuccessiveHalvingRunTest(unittest.TestCase):

    @staticmethod
    def obtain_config(data_point, param_indices):
        # data_points = list<mixed>
        # param_indices = dict<int, str>
        config = []
        for key in sorted(param_indices):
            config.append(data_point[key])
        return tuple(config)

    @staticmethod
    def check_sh_iteration(data_points, param_indices, eval_idx, file=None):
        # data_points = list<list<mixed>>
        # param_indices = dict<int, str>

        num_steps = int(math.log(len(data_points), 2))

        # enumerates over trace backwards, checking whether
        # - configs in a step also appeared in the previous step
        # - these configs were indeed amongst the best half
        # - the last one was selected as the best

        next_step_configs = {VerifySuccessiveHalvingRunTest.obtain_config(data_points[-1], param_indices)}

        for step in range(num_steps):
            current_configs = []
            current_scores = []
            for arms in range(2**(step+1), 2**(step+2)):
                if data_points[-arms][4] != 'false':
                    raise ValueError('Wrong incumbent')
                current_configs.append(VerifySuccessiveHalvingRunTest.obtain_config(data_points[-arms], param_indices))
                current_scores.append(float(data_points[-arms][eval_idx]))

            possible_continue_arms = set()
            current_scores = np.array(current_scores, dtype=np.float)
            sorted = np.argsort(current_scores * -1)
            num_continue_arms = int(len(current_configs) / 2)
            for i in range(len(current_configs)):
                if i < num_continue_arms or current_scores[sorted[i]] == current_scores[sorted[num_continue_arms-1]]:
                    possible_continue_arms.add(current_configs[sorted[i]])

            for config in next_step_configs:
                if config not in current_configs:
                    raise ValueError('Could not find config %s for file %s' %(str(config), file))

            if len(next_step_configs - possible_continue_arms) > 0:
                raise ValueError('Not correct arms continued. ')

            next_step_configs = set(current_configs)

    @staticmethod
    def check_hyperband_iteration(current_points, param_indices, eval_index, num_brackets, file):
        if num_brackets is None:
            VerifySuccessiveHalvingRunTest.check_sh_iteration(current_points, param_indices, eval_index, file)
        else:
            # this only handles 'vanilla' hyperband
            for i in range(num_brackets):
                num_data_points = 2 ** (num_brackets - i) - 1
                current_bracket_points = current_points[:num_data_points]
                current_points = current_points[num_data_points:]
                VerifySuccessiveHalvingRunTest.check_sh_iteration(current_bracket_points, param_indices, eval_index, file)

    @staticmethod
    def process_arff_file(file, num_brackets=None):
        arff_data = arff.load(open(file, 'r'))

        param_indices = dict()
        eval_index = None
        for idx, attribute in enumerate(arff_data['attributes']):
            if attribute[0].startswith('parameter_'):
                param_indices[idx] = attribute[0]
            elif attribute[0] == 'evaluation':
                eval_index = idx

        if len(param_indices) < 5:
            raise ValueError()

        # assumes order in the trace file..
        current_repeat = 0
        current_fold = 0
        current_points = []
        for datapoint in arff_data['data']:
            repeat = int(datapoint[0])
            fold = int(datapoint[1])
            if repeat != current_repeat or fold != current_fold:
                print('Checking %d %d with %d curves' % (repeat, fold, len(current_points)))
                VerifySuccessiveHalvingRunTest.check_hyperband_iteration(current_points, param_indices, eval_index, num_brackets, file)

                current_repeat = repeat
                current_fold = fold
                current_points = []
            current_points.append(datapoint)
        # verify the last batch
        VerifySuccessiveHalvingRunTest.check_hyperband_iteration(current_points, param_indices, eval_index, num_brackets, file)

    @staticmethod
    def traverse_experiment_directory(result_directory, num_brackets):
        for classifier in os.listdir(result_directory):
            if os.path.isfile(os.path.join(result_directory, classifier)):
                continue
            for fixed_parameters in os.listdir(os.path.join(result_directory, classifier)):
                print(openmlpimp.utils.get_time(), 'classifier:', classifier, fixed_parameters)
                directory = os.path.join(result_directory, classifier, fixed_parameters)

                for strategy in os.listdir(directory):
                    for task_directory in os.listdir(os.path.join(directory, strategy)):
                        file = os.path.join(directory, strategy, task_directory, 'trace.arff')

                        if os.path.isfile(file):
                            VerifySuccessiveHalvingRunTest.process_arff_file(file, num_brackets=num_brackets)

    def test_correct_successive_halving(self):
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/hyperband/')
        VerifySuccessiveHalvingRunTest.process_arff_file(os.path.join(directory, 'successive_halving_correct.arff'))
        pass

    def test_incorrect_successive_halving(self):
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/hyperband/')
        files = ['successive_halving_incorrect-1.arff',
                 'successive_halving_incorrect-2.arff',
                 'successive_halving_incorrect-3.arff',
                 'successive_halving_incorrect-4.arff']

        for file in files:
            with self.assertRaises(ValueError):
                VerifySuccessiveHalvingRunTest.process_arff_file(os.path.join(directory, file))

    def test_results_directory_sh(self):
        result_directory = os.path.expanduser('~') + '/nemo/experiments/20180206priorbased_experiments/'
        VerifySuccessiveHalvingRunTest.traverse_experiment_directory(result_directory, None)

    def test_results_directory_hyperband(self):
        result_directory = os.path.expanduser('~') + '/nemo/experiments/priorbased_experiments/'
        VerifySuccessiveHalvingRunTest.traverse_experiment_directory(result_directory, 5)