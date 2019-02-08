import argparse
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import os
import pandas as pd


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fanova_result_file',
                        default=os.path.expanduser('~/experiments/openml-pimp/fanova_adaboost_depth_1.csv'),
                        type=str)
    parser.add_argument('--output_directory', default=os.path.expanduser('~/experiments/openml-pimp'), type=str)
    parser.add_argument('--measure', default='predictive_accuracy', type=str)
    args_, misc = parser.parse_known_args()

    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    df = pd.read_csv(args.fanova_result_file)
    medians = df.groupby('hyperparameter')['importance_variance', 'importance_max_min'].median()
    print(df.dtypes)
    print(medians.dtypes)
    df = df.join(medians, on='hyperparameter', how='left', rsuffix='median_')
    print(df)


if __name__ == '__main__':
    run(read_cmd())
