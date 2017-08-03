import traceback
import openmlpimp
import datetime
import inspect
import json
import logging
import os
import sys
import time
import numpy as np
from matplotlib import pyplot as plt

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from fanova.fanova import fANOVA as fanova_pyrfr
from fanova.visualizer import Visualizer

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)


def read_cmd():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", default=12345, type=int,
                        help="random seed")
    parser.add_argument("-V", "--verbose_level", default=logging.INFO,
                        choices=["INFO", "DEBUG"],
                        help="verbosity")
    parser.add_argument("-C", "--table", action='store_true',
                        help="Save result table")
    parser.add_argument("-R", "--required_setups", default=200,
                        help="Minimum number of setups needed to use a task")
    parser.add_argument("-F", "--flow_id", default=6970,
                        help="The OpenML flow id to use")
    parser.add_argument("-T", "--openml_tag", default="study_14",
                        help="The OpenML tag for obtaining tasks")
    parser.add_argument('-P', '--fixed_parameters', type=json.loads, default=None,
                        help='Will only use configurations that have these parameters fixed')
    parser.add_argument('-L', '--logscale_parameters', type=json.loads,
                        default={'learning_rate': ''},
                        help='Parameters that are on a logscale')
    parser.add_argument('-I', '--ignore_parameters', type=json.loads,
                        default={'random_state': ''},
                        help='Parameters to ignore')
    parser.add_argument('-Q', '--use_quantiles', action="store_true",
                        default=True,
                        help='Use quantile information instead of full range')

    args_, misc = parser.parse_known_args()

    if args_.logscale_parameters is not None:
        args_.logscale_parameters = set(args_.logscale_parameters.keys())
    if args_.ignore_parameters is not None:
        args_.ignore_parameters = set(args_.ignore_parameters.keys())

    return args_


def plot_result(fANOVA, configspace, directory):
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
            with open(os.path.join(save_folder, 'pimp_values_fanova.json'), 'w') as out_file:
                json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
            # call plotting fn
            plot_result(evaluator, configspace, save_folder + "/fanova")
            return
        except ZeroDivisionError as e:
            pass
    raise e


if __name__ == '__main__':
    args = read_cmd()

    logging.basicConfig(level=args.verbose_level)
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    cache_folder = '/home/vanrijn/experiments/PIMP_flow%d_cache' %args.flow_id
    save_folder = '/home/vanrijn/experiments/PIMP_flow%d_%s' % (args.flow_id, ts)

    all_tasks = openmlpimp.utils.list_tasks(tag=args.openml_tag)
    print("Tasks: ", list(all_tasks.keys()), "(%d)" %len(all_tasks))

    total_ranks = None
    all_ranks = {}
    nr_tasks = 0
    for task_id in all_tasks:
        try:
            task_save_folder = save_folder + "/" + str(task_id)
            task_cache_folder = cache_folder + "/" + str(task_id)
            runhistory, configspace = openmlpimp.utils.cache_runhistory_configspace(task_cache_folder, args.flow_id, task_id, args)

            execute(task_save_folder, runhistory, configspace)
            results_file = task_save_folder + '/pimp_values_fanova.json'
            with open(results_file) as result_file:
                data = json.load(result_file)
                all_ranks[task_id] = data
                ranks = openmlpimp.utils.rank_dict(data, True)
                nr_tasks += 1
                if total_ranks is None:
                    total_ranks = ranks
                else:
                    total_ranks = openmlpimp.utils.sum_dict_values(total_ranks, ranks)
                print("Task", task_id, ranks)
        except Exception as e:
            print('error while executing task %d' %(task_id))
            traceback.print_exc()
    total_ranks = openmlpimp.utils.divide_dict_values(total_ranks, nr_tasks)
    print("TOTAL RANKS:", total_ranks, "("+str(nr_tasks)+")")
    openmlpimp.utils.to_csv_unpivot(all_ranks, save_folder + '/ranks_plain.csv')
    openmlpimp.utils.to_csv_file(all_ranks, save_folder + '/ranks.csv')
    openmlpimp.utils.plot_nemenyi(total_ranks, nr_tasks, save_folder + "/nemenyi.pdf")
