import argparse
import collections
import csv
import openml
import openmlpimp
import operator
import os
import matplotlib.pyplot as plt

def parse_args():
    all_classifiers = ['adaboost', 'decision_tree', 'libsvm_svc', 'random_forest', 'sgd']
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--virtual_env', type=str, default=os.path.expanduser('~') + '/projects/pythonvirtual/plot2/bin/python', help='python virtual env for plotting')
    parser.add_argument('--scripts_dir', type=str, default=os.path.expanduser('~') + '/projects/plotting_scripts/scripts', help='directory to Katha\'s plotting scripts')
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/experiments/optimizers/priors/', help='the directory to load the experiments from')

    return parser.parse_args()


def strategy_name(full_name):
    return full_name.split('__')[0]


def obtain_complete_tasks(results, tasks, classifiers=None):
    if classifiers is None:
        classifiers = results.keys()

    complete_tasks = list()
    for task_id in tasks.keys():
        in_all = True
        for classifier in classifiers:
            for strategy in results[clf_params]:
                if task_id not in results[classifier][strategy]:
                    in_all = False
        if in_all:
            complete_tasks.append(task_id)
    return complete_tasks


def bar_plots(results, all_tasks, classifiers=None):
    if classifiers is None:
        classifiers = results.keys()

    current_tasks = obtain_complete_tasks(results, all_tasks, classifiers)
    all_tasks_sorted = collections.OrderedDict(sorted(all_tasks.items(), key=operator.itemgetter(1)))

    colors = ['red', 'blue', 'green', 'magenta']

    plt.figure()
    for idx, clf_params in enumerate(classifiers):
        current_strategies = sorted(list(results[clf_params].keys()))

        barplot_data = []
        barplotlabel_names = []

        for task_id in all_tasks_sorted:
            if task_id not in current_tasks:
                continue
            score0 = results[clf_params][current_strategies[0]][task_id]
            score1 = results[clf_params][current_strategies[1]][task_id]
            current_dif = score0 - score1
            barplot_data.append(current_dif)
            barplotlabel_names.append(taskid_datasetname[task_id])

        width = 1 / len(classifiers)
        x_pos = list(range(len(barplot_data)))
        for i in range(len(x_pos)):
            x_pos[i] += idx * width

        # creates the bar plot
        plt.bar(x_pos, barplot_data, width=width, color=colors[idx])

    plt.xticks(list(range(len(barplotlabel_names))), barplotlabel_names, fontsize=6, rotation=45, ha='right')
    plt.savefig(os.path.join(args.result_directory, '__'.join(classifiers) + '_bars.pdf'), figsize=(32, 48), bbox_inches='tight')
    plt.close()

def box_plots(results, all_tasks, clf_params):
    boxplotlabel_names = []
    boxplot_data = []

    tasks = obtain_complete_tasks(results, all_tasks, [clf_params])

    for strategy in results[clf_params]:
        current = []
        for task_id in tasks:
            current.append(results[clf_params][strategy][task_id])
        boxplot_data.append(current)
        boxplotlabel_names.append(strategy_name(strategy))

    # creates the box plot
    plt.figure()
    plt.boxplot(boxplot_data)
    plt.xticks(list(range(1, len(boxplotlabel_names) + 1)), boxplotlabel_names, fontsize=12, rotation=45, ha='right')
    plt.savefig(os.path.join(args.result_directory, clf_params + '_boxplot.png'), bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    args = parse_args()
    results = collections.defaultdict(lambda: collections.defaultdict(dict))
    all_tasks = dict()
    taskid_datasetname = dict()

    for classifier in os.listdir(args.result_directory):
        if os.path.isfile(os.path.join(args.result_directory, classifier)):
            continue
        for fixed_parameters in os.listdir(os.path.join(args.result_directory, classifier)):
            print(openmlpimp.utils.get_time(), 'classifier:', classifier, fixed_parameters)
            directory = os.path.join(args.result_directory, classifier, fixed_parameters, 'curves_avg')
            clf_params = classifier + '__' + fixed_parameters
            for strategy in os.listdir(directory):
                for taskfile in os.listdir(os.path.join(directory, strategy)):
                    file = os.path.join(directory, strategy, taskfile)
                    task_id = int(taskfile.split('.')[0])
                    if task_id not in all_tasks:
                        dataset = openml.tasks.get_task(task_id).get_dataset()
                        all_tasks[task_id] = int(float(dataset.qualities['NumberOfInstances']))
                        taskid_datasetname[task_id] = dataset.name

                    with open(file) as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            pass

                    # row now contains the last row of the csv file
                    results[clf_params][strategy][task_id] = float(row['evaluation'])

        box_plots(results, all_tasks, clf_params)
        bar_plots(results, all_tasks, clf_params)
    bar_plots(results, all_tasks, results.keys())