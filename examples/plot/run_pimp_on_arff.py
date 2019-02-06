import arff
import argparse
import openmlcontrib


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../KDD2018/data/arff/svc.arff', type=str)
    args_, misc = parser.parse_known_args()

    return args_


def run(args):
    with open(args.dataset_path, 'r') as fp:
        arff_dataset = arff.load(fp)
    data = openmlcontrib.meta.arff_to_dataframe(arff_dataset, None)
    print(data)
    pass


if __name__ == '__main__':
    run(read_cmd())
