import collections
import openml
import openmlpimp
import random

from openml.utils import _tag_entity

# tags runs and counts runtime

study_orig = openml.study.get_study('OpenML100')
measure_name = 'usercpu_time_millis'
flows = [(6969, 'random_forest'), (6970, 'adaboost'), (7707, 'libsvm_svc')]
tasks = study_orig.tasks
random.shuffle(flows)
random.shuffle(tasks)

TAG = 'study_71'

ids = []

runs = []
flow_runtime = collections.defaultdict(float)
task_runtime = collections.defaultdict(float)

for flow_id, flow_name in flows:
    runtime = 0.0

    configuration_space = openmlpimp.utils.get_config_space_casualnames(flow_name, None)
    hyperparameters = dict(configuration_space._hyperparameters.items())

    for task_id in tasks:
        print(openmlpimp.utils.get_time(), flow_id, task_id)
        try:
            runid_runs = openmlpimp.utils.obtain_all_runs(flow=[flow_id], task=[task_id])
        except openml.exceptions.OpenMLServerException as e:
            print(flow_id, task_id, e.message)
            continue

        for run_id in runid_runs:
            run = openml.runs.get_run(run_id)
            setup = openml.setups.get_setup(run.setup_id)

            if not openmlpimp.utils.setup_complies_to_config_space(setup, hyperparameters):
                continue

            if TAG not in run.tags:
                try:
                    _tag_entity('run', run_id, TAG)
                    print('tagged run %d' %run_id)
                except openml.exceptions.OpenMLServerException as e:
                    print('%d was already tagged' %run_id)

            # run.fold_evaluations is a dict mapping from eval measure to repeat id to fold id to value (float)
            if run.fold_evaluations is not None and measure_name in run.fold_evaluations:

                current_runtimes = []
                for repeat in run.fold_evaluations[measure_name].keys():
                    for fold in run.fold_evaluations[measure_name][repeat].keys():
                        current_runtimes.append(run.fold_evaluations[measure_name][repeat][fold])

                if len(current_runtimes) != 10:
                    print('warning at run ', run_id, str(current_runtimes))

                sum_of_folds = sum(current_runtimes)
                flow_runtime[run.flow_id] += sum_of_folds
                task_runtime[run.task_id] += sum_of_folds
                runs.append(sum_of_folds)

        print(openmlpimp.utils.get_time(), "intermediate:", sum(runs))
        print(openmlpimp.utils.get_time(), "intermediate:", len(runs))
        print(openmlpimp.utils.get_time(), flow_runtime)
        print(openmlpimp.utils.get_time(), task_runtime)

print(openmlpimp.utils.get_time(), "TOTAL MILLIES:", sum(runs))
print(openmlpimp.utils.get_time(), "TOTAL RUNS:", len(runs))
print(openmlpimp.utils.get_time(), flow_runtime)
print(openmlpimp.utils.get_time(), task_runtime)
