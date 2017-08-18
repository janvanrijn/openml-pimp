import traceback
import openmlpimp
import datetime
import inspect
import json
import logging
import os
import sys
import time
import openml

from ConfigSpace.io.pcs_new import read
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from openmlpimp.backend.fanova import FanovaBackend
from openmlpimp.backend.pimp import PimpBackend

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
    parser.add_argument("-R", "--required_setups", default=100,
                        help="Minimum number of setups needed to use a task")
    parser.add_argument("-F", "--flow_id", default=6952,
                        help="The OpenML flow id to use")
    parser.add_argument("--model_type", default="libsvm_svc")
    parser.add_argument("-T", "--openml_studyid", default="14",
                        help="The OpenML tag for obtaining tasks")
    parser.add_argument('-P', '--fixed_parameters', type=json.loads, default={'kernel': 'rbf'},
                        help='Will only use configurations that have these parameters fixed')
    parser.add_argument('-L', '--logscale_parameters', type=json.loads,
                        default={'learning_rate': ''},
                        help='Parameters that are on a logscale')
    parser.add_argument('-I', '--ignore_parameters', type=json.loads,
                        default={'random_state': '', 'sparse': '', 'verbose': ''},
                        help='Parameters to ignore')
    parser.add_argument('-Q', '--use_quantiles', action="store_true",
                        default=True,
                        help='Use quantile information instead of full range')
    parser.add_argument('-M', '--modus', type=str, choices=['ablation', 'fanova'],
                        default='fanova', help='Whether to use ablation or fanova')

    args_, misc = parser.parse_known_args()

    if args_.logscale_parameters is not None:
        args_.logscale_parameters = set(args_.logscale_parameters.keys())
    if args_.ignore_parameters is not None:
        args_.ignore_parameters = set(args_.ignore_parameters.keys())

    return args_


if __name__ == '__main__':
    args = read_cmd()

    logging.basicConfig(level=args.verbose_level)
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    cache_folder = '/home/vanrijn/experiments/PIMP_flow%d_cache' %args.flow_id
    save_folder = '/home/vanrijn/experiments/PIMP_flow%d_%s' % (args.flow_id, ts)

    study = openml.study.get_study(args.openml_studyid, 'tasks')
    print("Tasks: ", list(study.tasks), "(%d)" %len(study.tasks))

    total_ranks = None
    all_ranks = {}
    nr_tasks = 0
    for task_id in study.tasks:
        try:
            task_save_folder = save_folder + "/" + str(task_id)
            task_cache_folder = cache_folder + "/" + str(task_id)

            # TODO: make the default!
            runhistory_path, configspace_path = openmlpimp.utils.cache_runhistory_configspace(task_cache_folder,
                                                                                              args.flow_id,
                                                                                              task_id,
                                                                                              model_type=args.model_type,
                                                                                              reverse=False,
                                                                                              args=args)

            if total_ranks is None:
                with open(configspace_path) as configspace_file:
                    configspace = read(configspace_file)
                total_ranks = {param.name: 0 for param in configspace.get_hyperparameters()}

            if args.modus == 'fanova':
                print('Running FANOVA backend on task %d' %task_id)
                results_file = FanovaBackend.execute(task_save_folder, runhistory_path, configspace_path)
            else:
                print('Running PIMP backend [%s] on task %d' %(args.modus, task_id))
                results_file = PimpBackend.execute(task_save_folder, runhistory_path, configspace_path, modus=args.modus)

            with open(results_file) as result_file:
                data = json.load(result_file)

                # for pimp backend
                if 'ablation' in data:
                    data = data['ablation']
                    # remove obsolute fields
                    if '-source-' in data:
                        del data['-source-']
                    if '-target-' in data:
                        del data['-target-']
                    # add missing fields
                    for param in total_ranks.keys():
                        if param not in data:
                            data[param] = 0.0
                if 'fanova' in data:
                    data = data['fanova']

                all_ranks[task_id] = data
                ranks = openmlpimp.utils.rank_dict(data, reverse=True)
                total_ranks = openmlpimp.utils.sum_dict_values(total_ranks, ranks, allow_subsets=False)
                nr_tasks += 1
                print("Task", task_id, ranks)
        except Exception as e:
            print('error while executing task %d' %(task_id))
            traceback.print_exc()
    total_ranks = openmlpimp.utils.divide_dict_values(total_ranks, nr_tasks)
    print("TOTAL RANKS:", total_ranks, "("+str(nr_tasks)+")")
    openmlpimp.utils.to_csv_unpivot(all_ranks, save_folder + '/ranks_plain.csv')
    openmlpimp.utils.to_csv_file(all_ranks, save_folder + '/ranks.csv')
