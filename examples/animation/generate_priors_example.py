import argparse
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.patches
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
    parser.add_argument('--filename', type=str, default='kde_example')

    parser.add_argument('--top_per_task', type=int, default=10)
    parser.add_argument('--param_min', type=int, default=0)
    parser.add_argument('--param_max', type=int, default=20)
    parser.add_argument('--density_ymax', type=float, default=0.2)

    parser.add_argument('--intro_frames', type=int, default=60)
    parser.add_argument('--added_point_frames', type=int, default=10)
    parser.add_argument('--kde_resolution', type=int, default=50)
    parser.add_argument('--interval', type=int, default=1)

    # output video params
    parser.add_argument('--gif', action='store_true', default=False)
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--bitrate', type=int, default=1800)

    return parser.parse_args()


def num_frames():
    frames = [param for param in data_gen()]
    return len(frames)


def data_gen():
    # draw first frame
    yield (0, False, 0, False)
    # keep frame same for |intro_frames|
    for i in range(1, args.intro_frames):
        yield (0, False, 0, True)

    # add points to KDE
    for i in range(args.top_per_task * len(tasks)):
        yield (i, True, 0, False)
        # keep frame same for |added_point_frames|
        for j in range(1, args.added_point_frames):
            yield (i, True, 0, True)
    # draw KDE (quickly)
    for i in range(args.kde_resolution):
        yield (None, False, i, False)


def plot(params):
    kde_data_size, highlight_last, kde_interval, keep_same = params
    print(params)
    if keep_same:
        return
    plt.clf()

    def rand_jitter(arr, seed=0):
        """
        Param is a (n, 2) nd_array. adds random jitter to the array
        """
        np.random.seed(seed)
        random_arr = np.array([np.random.randn() for _ in range(len(arr))])
        return arr + random_arr * 0.01

    def get_n_best(array, n, highest):
        """
        array is a (n, 2) nd_array. selects the n best elements, based on the 2nd column
        """
        coef = 1.0
        if highest:
            coef = -1.0
        array = array[(array[:, 1] * coef).argsort()]
        result = array[0:n]
        assert len(result) == n
        return result

    # If the three integers are R, C, and P in order,
    # the subplot will take the Pth position on a grid with R rows and C columns.
    plot_coordinates = [(2, 3, 1), (2, 3, 2), (2, 3, 3)]
    kde_data = []

    for idx, (nrows, ncols, plot_number) in enumerate(plot_coordinates):
        # create sub image
        sub = fig.add_subplot(nrows, ncols, plot_number)
        # cast values to 2d array of size (n, 2)
        values = np.array(task_datapoints[tasks[idx]])
        sub.set_xlim(args.param_min, args.param_max)
        sub.set_ylim(min(values[:, 1] - scatter_margin), max(values[:, 1] + scatter_margin))
        # determine how many points will be colored red
        num_top_points = args.top_per_task
        if kde_data_size is not None:
            num_top_points = min(num_top_points, kde_data_size - len(kde_data))
            if num_top_points >= kde_data_size - len(kde_data):
                circle_current_last = True
            else:
                circle_current_last = False

        # get the top and worst points
        n_best = get_n_best(values, num_top_points, True)
        n_worst = get_n_best(values, len(values) - num_top_points, False)
        kde_data.extend(n_best[:, 0])
        # plot blue (bad points) red (best points)
        sub.scatter(n_worst[:, 0], n_worst[:, 1], color='blue')
        sub.scatter(n_best[:, 0], n_best[:, 1], color='red')

        # add a circle to annotate the last added point:
        if highlight_last and circle_current_last and len(n_best) > 0:
            plt.scatter(n_best[-1, 0], n_best[-1, 1], facecolors='none', edgecolors='black')

        # remove ticks
        sub.set_xticks([])
        sub.set_yticks([])
        # adds labels
        sub.set_xlabel('Hyperparameter value')
        # first subplot also has a y label
        if idx == 0:
            sub.set_ylabel('Performance')

        # add which dataset we are using
        text_xpos = sub.get_xlim()[0] + text_margin * (sub.get_xlim()[1] - sub.get_xlim()[0])
        text_ypos = sub.get_ylim()[0] + text_margin * (sub.get_ylim()[1] - sub.get_ylim()[0])
        sub.text(x=text_xpos, y=text_ypos, s=datasets[idx] + ' dataset', ha='left', va='bottom', fontsize=8)

    sub = fig.add_subplot(2, 1, 2)
    sub.text(x=(args.param_max - args.param_min) / 2,
             y=args.density_ymax - text_margin,
             s='Prior across datasets',
             horizontalalignment='center')
    sub.set_xlabel('Hyperparameter value')
    sub.set_ylabel('Probability')
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlim(args.param_min, args.param_max)
    sub.set_ylim(0, args.density_ymax)

    # calculate KDE
    if len(kde_data) > 5 and (kde_interval is None or kde_interval > 0):
        density = scipy.stats.gaussian_kde(kde_data)
        xs = np.linspace(args.param_min, args.param_max, args.kde_resolution)
        if kde_interval is not None:
            xs = xs[0:kde_interval]

        density.covariance_factor = lambda: .25
        density._compute_covariance()
        # plot KDE
        sub.plot(xs, density(xs))

    if len(kde_data) > 0:
        # plot scatters for initial data
        kde_input_x = rand_jitter(kde_data)
        kde_input_y = rand_jitter([0.02]*len(kde_data))
        sub.scatter(x=kde_input_x, y=kde_input_y, color='red', marker='+')
        # make last scatter black
        if highlight_last and len(kde_data) > 0:
            sub.scatter(x=kde_input_x[-1], y=kde_input_y[-1], s=75, facecolors='none', edgecolors='black')

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
    text_margin = 0.02
    scatter_margin = 0.05
    tasks = [3, 6, 11]
    datasets = ['kr-vs-kp', 'letter', 'balance-scale']

    task_datapoints = dict()
    for idx, task_id in enumerate(tasks):
        task_datapoints[task_id] = get_evaluations_for_task(6969, task_id, 'min_samples_leaf', 50)

    fig = plt.figure()
    np.random.seed(0)

    ani = matplotlib.animation.FuncAnimation(fig, plot, data_gen,
                                             blit=False, interval=args.interval, repeat=False, save_count=num_frames())
    if args.gif:
        ani.save(os.path.join(args.output_dir, args.filename + '.gif'), dpi=args.dpi, writer='imagemagick')
    else:
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=args.fps, metadata=dict(artist='Jan N. van Rijn and Frank Hutter'), bitrate=args.bitrate)
        ani.save(os.path.join(args.output_dir, args.filename + '.mp4'), dpi=args.dpi, writer=writer)
    print('Done')
