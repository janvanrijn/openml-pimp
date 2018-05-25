import argparse


def read_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("-R", "--required_setups", default=100, help="Minimal number of setups needed to use a task")
    parser.add_argument("-F", "--flow_id", default=7707, help="The OpenML flow id to use")
    parser.add_argument("-F", "--task_id", default=3, help="The OpenML task id to use")
    parser.add_argument('-T', '--n_trees', type=int, default=16)
    parser.add_argument('-M', '--modus', type=str, choices=['ablation', 'fanova'],
                        default='fanova', help='Whether to use ablation or fanova')
    parser.add_argument('-L', '--limit', type=int, default=None, help='Max runs per task (efficiency)')

    return parser


def run():
    


if __name__ == '__main__':
    run(read_cmd())
