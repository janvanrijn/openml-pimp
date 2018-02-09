import arff
import argparse
import collections
import csv
import matplotlib.pyplot as plt
import openml
import openmlpimp
import os
import random
import sklearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--virtual_env', type=str, default=os.path.expanduser('~') + '/projects/pythonvirtual/plot2/bin/python')
    parser.add_argument('--scripts_dir', type=str, default=os.path.expanduser('~') + '/projects/plotting_scripts/scripts')
    parser.add_argument('--result_directory', type=str, default=os.path.expanduser('~') + '/nemo/experiments/priorbased_experiments/')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/optimizers/priors/')

    parser.add_argument('--measure', type=str, default='predictive_accuracy')
    parser.add_argument('--study_id', type=str, default='OpenML100')
    parser.add_argument('--task_limit', type=int, default=None)
    parser.add_argument('--setup', type=str, default='hyperband_5')
    parser.add_argument('--seed', type=int, default=None, help='the seed (efficiency)')

    args = parser.parse_args()
    return args


def list_classifiers(directory):
    result = []
    for classifier in os.listdir(directory):
        for param_setting in os.listdir(directory + '/' + classifier):
            if param_setting == 'vanilla':
                name = classifier
            else:
                name = classifier + ' (' + param_setting.split('_')[1] + ')'
            result.append((name, classifier + '/' + param_setting))
    return result


# def get_score_from_xml(run_xml_location, measure):
#     with open(run_xml_location, 'r') as fp:
#         run = openml.runs.functions._create_run_from_xml(fp.read())
#         scores = []
#
#         for repeat in run.fold_evaluations[measure]:
#             for fold in run.fold_evaluations[measure][repeat]:
#                 scores.append(float(run.fold_evaluations[measure][repeat][fold]))
#
#     return sum(scores) / len(scores)


# def get_score_from_avgcurve(csv_location):
#     with open(csv_location, 'r') as fp:
#         reader = csv.DictReader(fp)
#         for row in reader:
#             score = float(row['evaluation'])
#     return score


def trace_to_score(trace_file):
    with open(trace_file, 'r') as fp:
        trace_arff = arff.load(fp)
    trace = openml.runs.functions._create_trace_from_arff(trace_arff)

    curves_total = collections.defaultdict(float)
    curves_count = collections.defaultdict(int)
    for itt in trace.trace_iterations:
        cur = trace.trace_iterations[itt]
        curves_total[itt] += cur.evaluation
        curves_count[itt] += 1
    if len(set(curves_count.values())) != 1:
        # all curves should have same amount of subcurves (folds * repeats)
        raise ValueError()
    curves_avg = dict()
    for idx in curves_total:
        curves_avg[idx] = curves_total[idx] / curves_count[idx]

    # now return the max value:
    return max(curves_avg.values())


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


def plot_boxplot(priors_results, mode):
    output_directory = os.path.join(args.output_directory, args.setup)
    all = []
    keys = []
    for idx, classifier in enumerate(priors_results.keys()):
        data = []
        kde_wins = 0
        uni_wins = 0
        draws = 0
        for task_id in priors_results[classifier]:
            if len(priors_results[classifier][task_id]) == 2 and 'uniform' in priors_results[classifier][task_id] and 'kde' in priors_results[classifier][task_id]:
                scores_kde = priors_results[classifier][task_id]['kde']
                scores_uniform = priors_results[classifier][task_id]['uniform']

                current = sum(scores_kde.values()) / len(scores_kde) - sum(scores_uniform.values()) / len(scores_kde)
                data.append(current)
                if current > 0: kde_wins +=1
                elif current < 0: uni_wins += 1
                else: draws += 1
        all.append(data)
        keys.append(classifier)
        print(openmlpimp.utils.get_time(), mode, "%s kde %d vs %d uniform (and %d draws)" %(classifier, kde_wins, uni_wins, draws))

        plt.figure(figsize=(4, 12))
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.plot([0.5, 1.5], [0, 0], 'k-', linestyle='--', lw=1)
        plt.violinplot(data)
        plt.savefig(output_directory + '/priors_%s_%s.pdf' %(classifier, mode), bbox_inches='tight')
        plt.close()

    plt.figure()
    plt.violinplot(all, list(range(len(all))))
    plt.plot([-0.5, len(all)-0.5], [0, 0], 'k-', linestyle='--', lw=1)
    plt.xticks(list(range(len(keys))), keys, rotation=45, ha='right')
    output_file = output_directory + '/priors_violin_%s.pdf' %mode
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    study = openml.study.get_study(args.study_id, 'tasks')
    all_tasks = study.tasks
    if args.task_limit:
        all_tasks = random.sample(all_tasks, args.task_limit)

    all_classifiers = list_classifiers(args.result_directory + '/' + args.setup)
    print(openmlpimp.utils.get_time(), all_classifiers)
    priors_results_test = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    priors_results_valid = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    defaults_results = collections.defaultdict(dict)
    for classifier, directory_suffix in all_classifiers:
        for task_id in all_tasks:

            # default_run_directory = args.defaults_directory + directory_suffix + '/' + str(task_id)
            # default_run_xml = default_run_directory + '/run.xml'
            # if os.path.isfile(default_run_xml):
            #     default_score = get_score_from_xml(default_run_xml, args.measure)
            #     defaults_results[classifier][task_id] = default_score
            classifier_dir = os.path.join(args.result_directory, args.setup, directory_suffix)
            for strategy in os.listdir(classifier_dir):
                task_dir = os.path.join(classifier_dir, strategy, str(task_id))
                if os.path.isdir(task_dir):
                    for seed in os.listdir(task_dir):
                        if args.seed is not None and args.seed is not int(seed):
                            continue
                        strategy_name = strategy.split('__')[0]
                        strategy_trace = classifier_dir + '/' + strategy + '/' + str(task_id) + '/' + seed + '/trace.arff'
                        strategy_predictions = classifier_dir + '/' + strategy + '/' + str(task_id) + '/' + seed + '/predictions.arff'

                        if not os.path.isfile(strategy_predictions) or not os.path.isfile(strategy_trace):
                            print(openmlpimp.utils.get_time(), '%s: Task %d not finished for strategy %s seed %s' %(classifier, task_id, strategy, seed))
                            continue

                        with open(strategy_predictions, 'r') as fp:
                            predictions_arff = arff.load(fp)

                        run = openml.runs.OpenMLRun(flow_id=-1, dataset_id=-1, task_id=task_id)
                        run.data_content = predictions_arff['data']
                        score = run.get_metric_fn(sklearn.metrics.accuracy_score)

                        priors_results_test[classifier][task_id][strategy_name][seed] = score.mean()
                        priors_results_valid[classifier][task_id][strategy_name][seed] = trace_to_score(strategy_trace)
    # plot_scatter(defaults_results, priors_results_test)
    plot_boxplot(priors_results_test, 'test')
    plot_boxplot(priors_results_valid, 'validation')
