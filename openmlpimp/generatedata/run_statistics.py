import openml
import os


directory = os.path.expanduser('~') + '/nemo/experiments/rs_experiments'
measure = 'usercpu_time_millis'
runs = []

for classifier in os.listdir(directory):
    for fixed_parameters in os.listdir(os.path.join(directory, classifier)):
        print(classifier, fixed_parameters)
        for unoptimized_parameter in os.listdir(os.path.join(directory, classifier, fixed_parameters)):

            for unoptimized_parameter_value in os.listdir(os.path.join(directory, classifier, fixed_parameters, unoptimized_parameter)):

                for task_id in os.listdir(os.path.join(directory, classifier, fixed_parameters, unoptimized_parameter, unoptimized_parameter_value)):

                    file = os.path.join(directory, classifier, fixed_parameters, unoptimized_parameter, unoptimized_parameter_value, task_id, 'run.xml')

                    if os.path.isfile(file):

                        with open(file, 'r') as fp:

                            run_xml = fp.read()

                        run = openml.runs.functions._create_run_from_xml(run_xml)

                        runtimes = []

                        for repeat in run.fold_evaluations[measure]:
                            for fold in run.fold_evaluations[measure][repeat]:
                                runtimes.append(run.fold_evaluations[measure][repeat][fold])
                        runs.append(sum(runtimes))

print(len(runs))
print(sum(runs))