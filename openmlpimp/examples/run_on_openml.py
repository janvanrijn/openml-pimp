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

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pimp.importance.importance import Importance
from ConfigSpace.io.pcs_new import write

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

def read_cmd():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-M", "--modus",
                        help='Analysis method to use', default="fanova",
                        choices=['ablation', 'forward-selection', 'influence-model', 'fanova'])
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


def generate_required_files(folder, flow_id, task_id,
                            required_setups=None,
                            fixed_parameters=None,
                            logscale_parameters=None,
                            ignore_parameters=None):
    try:
      os.makedirs(folder)
    except FileExistsError:
      pass
    runhistory, configspace = openmlpimp.utils.obtain_runhistory_and_configspace(flow_id, task_id,
                                                                                 required_setups=required_setups,
                                                                                 fixed_parameters=fixed_parameters,
                                                                                 logscale_parameters=logscale_parameters,
                                                                                 ignore_parameters=ignore_parameters)

    trajectory_lines = openmlpimp.utils.runhistory_to_trajectory(runhistory, None)

    with open(folder + 'runhistory.json', 'w') as outfile:
        json.dump(runhistory, outfile)
        runhistory_location = os.path.realpath(outfile.name)

    with open(folder + 'traj_aclib2.json', 'w') as outfile:
        for line in trajectory_lines:
            json.dump(line, outfile)
            outfile.write("\n")
        traj_location = os.path.realpath(outfile.name)

    with open(folder + 'config_space.pcs', 'w') as outfile:
        outfile.write(write(configspace))
        pcs_location = os.path.realpath(outfile.name)

    with open(folder + 'scenario.txt', 'w') as outfile:
        outfile.write("run_obj = quality\ndeterministic = 1\nparamfile = " + pcs_location)
        scenario_location = os.path.realpath(outfile.name)

    return scenario_location, runhistory_location, traj_location


def execute(save_folder, flow_id, task_id, args):

    scenario, runhistory, trajectory = generate_required_files(save_folder + '/inputs/',
                                                               flow_id, task_id,
                                                               required_setups=args.required_setups,
                                                               fixed_parameters=args.fixed_parameters,
                                                               logscale_parameters=args.logscale_parameters,
                                                               ignore_parameters=args.ignore_parameters)
    importance = Importance(scenario, runhistory,
                            traj_file=trajectory,
                            seed=args.seed,
                            save_folder=save_folder,
                            cutoffs_rf=cutoffs_rf)
    for i in range(5):
        try:
            result = importance.evaluate_scenario(args.modus)

            with open(os.path.join(save_folder, 'pimp_values_%s.json' % args.modus), 'w') as out_file:
                json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
            importance.plot_results(name=os.path.join(save_folder, args.modus))
            return
        except ZeroDivisionError as e:
            pass
    raise e


if __name__ == '__main__':
    args = read_cmd()

    logging.basicConfig(level=args.verbose_level)
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    save_folder = '/home/vanrijn/experiments/PIMP_flow%d_%s_%s' % (args.flow_id, args.modus, ts)

    all_tasks = openmlpimp.utils.list_tasks(tag=args.openml_tag)
    print("Tasks: ", str(all_tasks.keys()), "(%d)" %len(all_tasks))

    total_ranks = None
    all_ranks = {}
    nr_tasks = 0
    for task_id in all_tasks:
        try:
            task_folder = save_folder + "/" + str(task_id)
            execute(task_folder, args.flow_id, task_id, args)
            results_file = save_folder + '/' + str(task_id) + '/pimp_values_%s.json' %args.modus
            with open(results_file) as result_file:
                data = json.load(result_file)
                all_ranks[task_id] = data[args.modus]
                ranks = openmlpimp.utils.rank_dict(data[args.modus], True)
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
