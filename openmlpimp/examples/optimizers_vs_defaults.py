import argparse
import collections
import csv
import json
import matplotlib.pyplot as plt
import openml
import openmlpimp
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--virtual_env', type=str, default=os.path.expanduser('~') + '/projects/pythonvirtual/plot2/bin/python')
    parser.add_argument('--scripts_dir', type=str, default=os.path.expanduser('~') + '/projects/plotting_scripts/scripts')
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/experiments/optimizers/priors/')
    parser.add_argument('--defaults_directory', type=str, default=os.path.expanduser('~') + '/experiments/defaults/')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/optimizers/priors/')

    parser.add_argument('--measure', type=str, default='predictive_accuracy')
    parser.add_argument('--study_id', type=str, default=14)

    args = parser.parse_args()
    return args


def list_classifiers(directory):
    result = []
    for classifier in os.listdir(directory):
        for param_setting in os.listdir(directory + '/' + classifier):
            print(param_setting)
            if param_setting == 'vanilla':
                name = classifier
            else:
                name = classifier + ' (' + param_setting.split('_')[1] + ')'
            result.append((name, classifier + '/' + param_setting))
    return result


def get_score_from_xml(run_xml_location, measure):
    with open(run_xml_location, 'r') as fp:
        run = openml.runs.functions._create_run_from_xml(fp.read())
        scores = []

        for repeat in run.fold_evaluations[measure]:
            for fold in run.fold_evaluations[measure][repeat]:
                scores.append(float(run.fold_evaluations[measure][repeat][fold]))

    return sum(scores) / len(scores)


def get_score_from_avgcurve(csv_location):
    with open(csv_location, 'r') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            score = float(row['evaluation'])
    return score


def plot_scatter(defaults_results, priors_results):
    colors = ['r', 'b', 'g', 'c']

    # print("task_id default  uniform  kde")
    plt.figure()
    plt.plot([-1, 1], [-1, 1], 'k-', linestyle='-', lw=1)
    for idx, classifier in enumerate(priors_results.keys()):
        plot_data_x = []
        plot_data_y = []
        for task_id in priors_results[classifier]:
            if len(priors_results[classifier][task_id]) == 2 and 'uniform' in priors_results[classifier][task_id] and 'kde' in priors_results[classifier][task_id] and task_id in defaults_results[classifier]:
                # print("%7d %7f %7f %7f" %(task_id, defaults_results[classifier][task_id], priors_results[classifier][task_id]['uniform'], priors_results[classifier][task_id]['kde']))
                plot_data_x.append(priors_results[classifier][task_id]['uniform'] - defaults_results[classifier][task_id])
                plot_data_y.append(priors_results[classifier][task_id]['kde'] - defaults_results[classifier][task_id])
        plt.scatter(plot_data_x, plot_data_y, color=colors[idx], label=classifier)
    # print("(%5d) %7f %7f %7f" %(len(plot_data_x), 0, sum(plot_data_x) / len(plot_data_x), sum(plot_data_y) / len(plot_data_y)))

    plt.legend(loc='upper left')
    plt.xlim((-0.25, 0.8))
    plt.ylim((-0.25, 0.8))
    plt.savefig('/home/vanrijn/experiments/priors_scatter.pdf')
    plt.close()


def plot_boxplot(priors_results):
    all = []
    keys = []
    for idx, classifier in enumerate(priors_results.keys()):
        data = []
        for task_id in priors_results[classifier]:
            if len(priors_results[classifier][task_id]) == 2 and 'uniform' in priors_results[classifier][task_id] and 'kde' in priors_results[classifier][task_id]:
                data.append(priors_results[classifier][task_id]['kde'] - priors_results[classifier][task_id]['uniform'])
        all.append(data)
        keys.append(classifier)

        plt.figure(figsize=(4,12))
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.plot([0.5, 1.5], [0, 0], 'k-', linestyle='--', lw=1)
        plt.violinplot(data)
        plt.savefig('/home/vanrijn/experiments/priors_%s.pdf' %classifier, bbox_inches='tight')
        plt.close()

    plt.figure()
    plt.violinplot(all, list(range(len(all))))
    plt.plot([-0.5, len(all)-0.5], [0, 0], 'k-', linestyle='--', lw=1)
    plt.xticks(list(range(len(keys))), keys, rotation=45, ha='right')
    plt.savefig('/home/vanrijn/experiments/priors_violin.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    study = openml.study.get_study(args.study_id, 'tasks')

    all_classifiers = list_classifiers(args.result_directory)
    print(all_classifiers)
    priors_results = collections.defaultdict(lambda: collections.defaultdict(dict))
    defaults_results = collections.defaultdict(dict)
    for classifier, directory_suffix in all_classifiers:
        for task_id in study.tasks:

            default_run_directory = args.defaults_directory + directory_suffix + '/' + str(task_id)
            default_run_xml = default_run_directory + '/run.xml'
            if os.path.isfile(default_run_xml):
                default_score = get_score_from_xml(default_run_xml, args.measure)
                defaults_results[classifier][task_id] = default_score

            for strategy in os.listdir(args.result_directory + directory_suffix + '/curves_avg'):
                strategy_name = strategy.split('__')[0]
                strategy_trace = args.result_directory + directory_suffix + '/curves_avg/' + strategy + '/' + str(task_id) + '.csv'
                if not os.path.isfile(strategy_trace):
                    print('Task %d not finished for strategy %s' %(task_id, strategy))
                    continue

                score = get_score_from_avgcurve(strategy_trace)
                priors_results[classifier][task_id][strategy_name] = score
    plot_scatter(defaults_results, priors_results)
    plot_boxplot(priors_results)
