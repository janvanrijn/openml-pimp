import argparse
import os
import openmlpimp

# CMD: sshfs fr_jv1031@login1.nemo.uni-freiburg.de:/home/fr/fr_fr/fr_jv1031 nemo/
plotting_virtual_env = '/home/vanrijn/projects/pythonvirtual/plot2/bin/python'
plotting_scripts_dir = '/home/vanrijn/projects/plotting_scripts/scripts'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--directory', type=str, default=os.path.expanduser('~') + '/nemo/experiments/random_search_prior/libsvm_svc/kernel_poly', help='the directory to load the experiments from')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    output_directory = '/home/vanrijn/experiments/optimizers/priors/'
    args = parse_args()

    all_taskids = set()

    all_strategies = os.listdir(args.directory)
    strategy_directories = {}
    for strategy in all_strategies:
        directory = os.path.join(args.directory, strategy)
        strategy_directories[strategy] = output_directory + strategy + '/'
        task_directories = os.listdir(directory)
        for task in task_directories:
            arff_file = os.path.join(args.directory, strategy, task, 'trace.arff')
            if os.path.isfile(arff_file):
                openmlpimp.utils.obtain_performance_curves_arff(arff_file, output_directory + strategy + '/' + task + '/')
                all_taskids.add(task)

    for task_id in all_taskids:
        openmlpimp.utils.plot_task(plotting_virtual_env, plotting_scripts_dir, strategy_directories, output_directory + 'plots/', task_id)

