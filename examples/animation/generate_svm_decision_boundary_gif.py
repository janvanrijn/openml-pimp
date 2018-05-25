import argparse
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import os
import sklearn.svm
import sklearn.datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Generate svm boundary gif')
    # output file related
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/pimp/movies')
    parser.add_argument('--filename', type=str, default='svm_decision_boundary')

    # input dataset related
    parser.add_argument('--num_datapoints', type=int, default=400)
    parser.add_argument('--data_factor', type=float, default=.2)
    parser.add_argument('--data_noise', type=float, default=.25)

    # gamma hyperparameter related
    parser.add_argument('--param_min', type=int, default=-3)
    parser.add_argument('--param_max', type=int, default=8)
    parser.add_argument('--param_interval', type=float, default=0.02)

    # plot
    parser.add_argument('--plot_margin', type=int, default=0.1)
    parser.add_argument('--interval', type=int, default=10)

    # output video params
    parser.add_argument('--gif', action='store_true', default=False)
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--bitrate', type=int, default=1800)

    return parser.parse_args()


def num_frames():
    return int(np.floor((args.param_max - args.param_min) / args.param_interval))


def data_gen():
    for gamma in np.arange(args.param_min, args.param_max, args.param_interval):
        yield (gamma, 1)


def make_meshgrid(x, y, margin, delta=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - margin, x.max() + margin
    y_min, y_max = y.min() - margin, y.max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, delta),
                         np.arange(y_min, y_max, delta))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_svm(params):
    print(params)
    gamma_exp, C_exp = params

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    model = sklearn.svm.SVC(kernel='rbf', gamma=2.0**gamma_exp, C=2.0**C_exp)
    model.fit(X, y)
    performance = sklearn.metrics.accuracy_score(y, model.predict(X))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, args.plot_margin)
    ax.clear()
    plot_contours(ax, model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.set_title('Support Vector Machine')
    ax.text(x=xx[0, 0] + args.plot_margin, y=yy[-1, -1] - args.plot_margin,
            s='Gamma: $2^{%0.2f}$\nComplexity: $2^{%0.0f}$\nAccuracy: %0.2f' % (gamma_exp, C_exp, performance),
            verticalalignment='top', horizontalalignment='left')

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='black')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')


if __name__ == '__main__':
    args = parse_args()

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)
    np.random.seed(0)

    X, y = sklearn.datasets.make_circles(n_samples=args.num_datapoints, factor=args.data_factor, noise=args.data_noise)

    ani = matplotlib.animation.FuncAnimation(fig, plot_svm, data_gen,
                                             blit=False, interval=args.interval, repeat=False, save_count=num_frames())
    os.makedirs(args.output_dir, exist_ok=True)
    if args.gif:
        ani.save(os.path.join(args.output_dir, args.filename + '.gif'), dpi=args.dpi, writer='imagemagick')
    else:
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=args.fps, metadata=dict(artist='Jan N. van Rijn and Frank Hutter'), bitrate=args.bitrate)
        ani.save(os.path.join(args.output_dir, args.filename + '.mp4'), dpi=args.dpi, writer=writer)
    print('Done')
