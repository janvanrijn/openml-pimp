import argparse
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import os
import openml
import pickle
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description='Generate svm boundary gif')
    # output file related
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/pimp/movies')
    parser.add_argument('--cache_dir', type=str, default=os.path.expanduser('~') + '/experiments/pimp/cache')
    parser.add_argument('--filename', type=str, default='svm_decision_boundary')

    parser.add_argument('--top_per_task', type=int, default=10)
    parser.add_argument('--param_min', type=int, default=0)
    parser.add_argument('--param_max', type=int, default=20)

    # output video params
    parser.add_argument('--gif', action='store_true', default=False)
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--bitrate', type=int, default=1800)

    return parser.parse_args()


def plot():
    def rand_jitter(arr):
        stdev = .01 * (max(arr) - min(arr)) + 0.005
        return arr + np.random.randn(len(arr)) * stdev

    def get_n_best(array, n, highest):
        array = array[array[:, 1].argsort()]
        if highest:
            result = array[len(array)-n:]
        else:
            result = array[0:n]
        assert len(result) == n
        return result

    fig = plt.figure()

    # If the three integers are R, C, and P in order,
    # the subplot will take the Pth position on a grid with R rows and C columns.
    plot_coordinates = [(2, 3, 1), (2, 3, 2), (2, 3, 3)]
    kde_data = []

    for idx, (nrows, ncols, plot_number) in enumerate(plot_coordinates):
        sub = fig.add_subplot(nrows, ncols, plot_number)
        values = np.array(task_datapoints[tasks[idx]])
        n_best = get_n_best(values, args.top_per_task, True)
        n_worst = get_n_best(values, len(values) - args.top_per_task, False)
        kde_data.extend(n_best[:, 0])
        sub.scatter(n_worst[:, 0], n_worst[:, 1], color='blue')
        sub.scatter(n_best[:, 0], n_best[:, 1], color='red')
        sub.set_xticks([])
        sub.set_yticks([])
        sub.set_xlabel('Hyperparameter value')
        if idx == 0:
            sub.set_ylabel('Performance')

    sub = fig.add_subplot(2, 1, 2)
    sub.set_xlabel('Hyperparameter value')
    sub.set_xticks([])
    sub.set_yticks([])

    density = scipy.stats.gaussian_kde(kde_data)
    xs = np.linspace(args.param_min, args.param_max, 200)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    sub.plot(xs, density(xs))

    sub.scatter(x=rand_jitter(kde_data), y=rand_jitter([0.01]*len(kde_data)), color='red', marker='+')
    sub.set_xlim(args.param_min, args.param_max)

    fig.tight_layout()


def get_evaluations_for_task(flow_id, task_id, hyperparameter, limit):
    os.makedirs(args.cache_dir, exist_ok=True)
    file_location = os.path.join(args.cache_dir, 'task_%d_flow_%d_param_%s_size_%d.pkl'
                                 % (task_id, flow_id, hyperparameter, limit))
    if os.path.isfile(file_location):
        with open(file_location, 'rb') as fp:
            return pickle.load(fp)

    evaluations = openml.evaluations.list_evaluations('predictive_accuracy', flow=[flow_id], task=[task_id], size=limit)
    setup_ids = [evaluation.setup_id for evaluation in evaluations.values()]

    setups = openml.setups.list_setups(flow=flow_id, setup=setup_ids)
    setup_hyperparametervalue = dict()
    for setup_id, setup in setups.items():
        for param in setup.parameters.values():
            if param.parameter_name == hyperparameter:
                setup_hyperparametervalue[setup_id] = param.value
        if setup_id not in setup_hyperparametervalue:
            raise ValueError('Could not find hyperparameter %s in setup: %d' % (hyperparameter, setup_id))

    res = [(int(setup_hyperparametervalue[eva.setup_id]), eva.value) for eva in evaluations.values()]
    with open(file_location, 'wb') as fp:
        pickle.dump(res, fp)
    return res


if __name__ == '__main__':
    args = parse_args()
    tasks = [3, 6, 11]

    task_datapoints = dict()
    for idx, task_id in enumerate(tasks):
        task_datapoints[task_id] = get_evaluations_for_task(6969, task_id, 'min_samples_leaf', 50)

    np.random.seed(0)
    plot()
    plt.show()
