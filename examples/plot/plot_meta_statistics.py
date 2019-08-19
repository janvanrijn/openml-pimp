import arff
import argparse
import logging
import matplotlib.pyplot as plt
import openmlcontrib
import os
import seaborn as sns


# to plot: <openml_pimp_root>/examples/plot/plot_fanova_aggregates.py
def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../DS2019/data/resnet.arff', type=str)
    parser.add_argument('--output_directory', default=os.path.expanduser('~/experiments/openml-pimp'), type=str)
    parser.add_argument('--y_column', default='predictive_accuracy', type=str)
    parser.add_argument('--task_id_column', default='dataset', type=str)
    parser.add_argument('--index_columns', nargs='+', type=str, default=[
        'batch_size', 'horizontal_flip', 'learning_rate_init', 'learning_rate_decay', 'momentum', 'patience',
        'resize_crop', 'shuffle', 'tolerance', 'vertical_flip', 'weight_decay', 'dataset'
    ])
    parser.add_argument('--selection_column', type=str, default='epochs')
    args_, misc = parser.parse_known_args()

    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    with open(args.dataset_path) as fp:
        arff_dict = arff.load(fp)
    df = openmlcontrib.meta.arff_to_dataframe(arff_dict)
    # max value for selection column (i.e., max number of epochs per dataset)
    print(df.groupby('dataset')[args.selection_column].max())
    df = df.loc[df.reset_index().groupby(args.index_columns)[args.selection_column].idxmax()]
    # number of results per dataset based on the selection column
    print(df.groupby('dataset')[args.y_column].count())
    # best performance per dataset
    print(df.groupby('dataset')[args.y_column].max())

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.boxplot(x=args.task_id_column, y=args.y_column, ax=ax, data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel(args.y_column.replace('_', ' ').capitalize())
    output_file = os.path.join(args.output_directory, 'results_%s.pdf' % args.y_column)
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)


if __name__ == '__main__':
    run(read_cmd())
