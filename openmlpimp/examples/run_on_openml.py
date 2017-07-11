import random
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

from argparse import  ArgumentDefaultsHelpFormatter, ArgumentParser
from pimp.importance.importance import Importance
from ConfigSpace.io.pcs_new import write


def read_cmd():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-M", "--modus",
                        help='Analysis method to use', default="ablation",
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

    args_, misc = parser.parse_known_args()

    return args_


def generate_required_files(folder, flow_id, task_id):
    runhistory, configspace = openmlpimp.utils.obtain_runhistory_and_configspace(flow_id, task_id)

    default_setup_id = random.sample((runhistory['configs'].keys()), 1)[0] # TODO
    trajectory_lines = openmlpimp.utils.runhistory_to_trajectory(runhistory, default_setup_id)

    with open('runhistory.json', 'w') as outfile:
        json.dump(runhistory, outfile)
        runhistory_location = os.path.realpath(outfile.name)

    with open('traj_aclib2.json', 'w') as outfile:
        for line in trajectory_lines:
            json.dump(line, outfile)
            outfile.write("\n")
        traj_location = os.path.realpath(outfile.name)

    with open('config_space.pcs', 'w') as outfile:
        outfile.write(write(configspace))
        pcs_location = os.path.realpath(outfile.name)

    with open('scenario.txt', 'w') as outfile:
        outfile.write("run_obj = quality\ndeterministic = 1\nparamfile = " + pcs_location)
        scenario_location = os.path.realpath(outfile.name)

    return scenario_location, runhistory_location, traj_location

if __name__ == '__main__':
    scenario, runhistory, trajectory = generate_required_files('input/', 6969, 11)

    args = read_cmd()
    logging.basicConfig(level=args.verbose_level)
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    save_folder = 'PIMP_%s_%s' % (args.modus, ts)

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
