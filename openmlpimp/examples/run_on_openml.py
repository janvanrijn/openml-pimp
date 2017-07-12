import traceback
import openmlpimp
import datetime
import inspect
import json
import logging
import os
import sys
import time

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pimp.importance.importance import Importance
from ConfigSpace.io.pcs_new import write


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
    parser.add_argument("-I", "--impute", action='store_true',
                        help="Impute censored data")
    parser.add_argument("-C", "--table", action='store_true',
                        help="Save result table")
    parser.add_argument("-R", "--required_setups", default=30,
                        help="Minimum number of setups needed to use a task")
    parser.add_argument("-F", "--flow_id", default=6952,
                        help="The OpenML flow id to use")
    parser.add_argument("-T", "--openml_tag", default="study_14",
                        help="The OpenML tag for obtaining tasks")
    parser.add_argument('-N', '--n_instances', type=str, default='1..2000',
                        help='restrict obtained tasks to certain nr of instances, e.g., 1..1000')
    parser.add_argument('-P', '--fixed_parameters', type=json.loads, default='{"kernel": "rbf"}',
                        help='Will only use configurations that have these parameters fixed')

    args_, misc = parser.parse_known_args()

    return args_


def generate_required_files(folder, flow_id, task_id, required_setups=None, fixed_parameters=None):
    try:
      os.makedirs(folder)
    except FileExistsError:
      pass
    runhistory, configspace = openmlpimp.utils.obtain_runhistory_and_configspace(flow_id, task_id, required_setups=required_setups, fixed_parameters=fixed_parameters)

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
                                                               fixed_parameters=args.fixed_parameters)
    importance = Importance(scenario, runhistory,
                            traj_file=trajectory,
                            seed=args.seed,
                            save_folder=save_folder,
                            impute_censored=args.impute)  # create importance object
    save_folder += '_run1'
    with open(os.path.join(save_folder, 'pimp_args.json'), 'w') as out_file:
        json.dump(args.__dict__, out_file, sort_keys=True, indent=4, separators=(',', ': '))
    result = importance.evaluate_scenario(args.modus)

    with open(os.path.join(save_folder, 'pimp_values_%s.json' % args.modus), 'w') as out_file:
        json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
    importance.plot_results(name=os.path.join(save_folder, args.modus))


if __name__ == '__main__':
    args = read_cmd()
    logging.basicConfig(level=args.verbose_level)
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    save_folder = '/home/vanrijn/experiments/PIMP_flow%d_%s_%s' % (args.flow_id, args.modus, ts)

    all_tasks = openmlpimp.utils.list_tasks(tag=args.openml_tag, nr_instances=args.n_instances)\

    total_ranks = None
    for task_id in all_tasks:
        try:
            task_folder = save_folder + "/" + str(task_id)
            execute(task_folder, args.flow_id, task_id, args)
            results_file = save_folder + '/' + str(task_id) + '_run1/' + 'pimp_values_fanova.json'
            with open(results_file) as result_file:
                data = json.load(result_file)
                ranks = openmlpimp.utils.rank_dict(data['fanova'], True)
                if total_ranks is None:
                    total_ranks = ranks
                else:
                    total_ranks = openmlpimp.utils.sum_dict_values(total_ranks, ranks)
                print("Task", task_id, ranks)
        except Exception as e:
            print('error while executing task %d' %(task_id))
            traceback.print_exc()
    print("TOTAL RANKS:", total_ranks)