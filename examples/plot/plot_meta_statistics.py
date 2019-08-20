import arff
import argparse
import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openmlcontrib
import os
import seaborn as sns


# to plot: <openml_pimp_root>/examples/plot/plot_fanova_aggregates.py
def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../DS2019/data/resnet.arff', type=str)
    parser.add_argument('--output_directory', default=os.path.expanduser('~/experiments/openml-pimp'), type=str)
    parser.add_argument('--font_size', default=16, type=int)
    parser.add_argument('--y_column', default='predictive_accuracy', type=str)
    parser.add_argument('--task_id_column', default='dataset', type=str)
    parser.add_argument('--index_columns', nargs='+', type=str, default=[
        'batch_size', 'horizontal_flip', 'learning_rate_init', 'learning_rate_decay', 'momentum', 'patience',
        'resize_crop', 'shuffle', 'tolerance', 'vertical_flip', 'weight_decay', 'dataset'
    ])
    parser.add_argument('--selection_column', type=str, default='epochs')
    parser.add_argument('--dataset_name_map', type=json.loads, default='{"svhn": "SVHN", "stl10": "STL-10", '
                                                                       '"dvc": "Dog Vs. Cat", "cifar10": "Cifar-10", '
                                                                       '"cifar100": "Cifar-100", "fruits": "Fruits 360", '
                                                                       '"flower": "Flower", "mnist": "MNIST", '
                                                                       '"fmnist": "Fashion MNIST", "scmnist": "HAM10000"}')
    args_, misc = parser.parse_known_args()

    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # and do the plotting
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {
        'text.usetex': True,
        'font.size': args.font_size,
        'font.family': 'lmodern',
        'text.latex.unicode': True,
    }
    matplotlib.rcParams.update(params)

    with open(args.dataset_path) as fp:
        arff_dict = arff.load(fp)
    df = openmlcontrib.meta.arff_to_dataframe(arff_dict)
    df['dataset'] = df['dataset'].replace(args.dataset_name_map)

    # max value for selection column (i.e., max number of epochs per dataset)
    print(df.groupby('dataset')[args.selection_column].max())
    df = df.loc[df.reset_index().groupby(args.index_columns)[args.selection_column].idxmax()]
    # number of results per dataset based on the selection column
    print(df.groupby('dataset')[args.y_column].count())
    # best performance per dataset
    print(df.groupby('dataset')[args.y_column].max())

    medians = df.groupby('dataset')[args.y_column].median()
    df = df.join(medians, on='dataset', how='left', rsuffix='_median').sort_values('%s_median' % args.y_column)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.boxplot(x=args.task_id_column, y=args.y_column, ax=ax, data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel(args.y_column.replace('_', ' ').capitalize())
    ax.set_xlabel(None)
    output_file = os.path.join(args.output_directory, 'results_%s.pdf' % args.y_column)
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)


if __name__ == '__main__':
    run(read_cmd())
