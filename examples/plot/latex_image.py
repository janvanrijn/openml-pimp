import argparse
import os
import typing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, default=os.path.expanduser('~/experiments/openml-pimp/marginal_plots'))
    parser.add_argument('--image_width', type=str, default='.189\\textwidth')
    parser.add_argument('--default_extension', type=str, default='pdf')
    parser.add_argument('--images_per_page', type=int, default=7)
    parser.add_argument('--caption', type=str, default='Marginal plots for test accuracy per dataset')
    parser.add_argument('--hyperparameters', type=str, nargs='+',  default=[
        'epochs', 'learning_rate_init', 'weight_decay', 'momentum', 'epochs__learning_rate_init']
    )
    parser.add_argument('--datasets', type=str, nargs='+',  default=[
        'cifar10', 'cifar100', 'dvc', 'flower', 'fmnist', 'fruits', 'mnist', 'scmnist', 'stl10', 'svhn']
    )

    args = parser.parse_args()
    return args


def output_latex(latex: typing.List[str], caption: str, subimage_counter: int):
    if subimage_counter > 0:
        print('\\addtocounter{figure}{-1}')
        print('\\addtocounter{subfigure}{7}')
    print('\\begin{figure}[tp!]\n\t\\begin{center}')
    print('\n'.join(latex))
    print('\t\t\\caption{%s%s}' % (caption, ' cont\'d' if subimage_counter > 0 else ''))
    print('\t\\end{center}\n\\end{figure}')


def run(args):
    current_latex = []
    last_output = 0
    for d_idx, dataset in enumerate(args.datasets):
        if d_idx > 0 and d_idx % args.images_per_page == 0:
            output_latex(current_latex, args.caption, last_output)
            current_latex = list()
            last_output = d_idx

        dataset_latex = []
        for h_idx, hyperparameter_plot in enumerate(args.hyperparameters):
            relative_path = os.path.join('%s__%s.%s' % (dataset, hyperparameter_plot, args.default_extension))
            full_path = os.path.join(args.input_directory, relative_path)
            if not os.path.isfile(full_path):
                raise ValueError('Could not find plot: %s' % full_path)
            dataset_latex.append('\t\t\t\\includegraphics[width=%s]{images/marginal_plots/%s}' % (args.image_width, relative_path))
        current_latex.append('\t\t\\subfigure[%s] {\n%s\n\t\t}' % (dataset, '\n'.join(dataset_latex)))
    if len(current_latex) > 0:
        output_latex(current_latex, args.caption, last_output)


if __name__ == '__main__':
    run(parse_args())
