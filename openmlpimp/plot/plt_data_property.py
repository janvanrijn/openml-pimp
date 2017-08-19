import matplotlib.pyplot as plt
import openml
import csv

from collections import OrderedDict

x_axis_feature = 'NumberOfInstances'
x_axis_label = 'Number of Data Points'

y_axis_feature = 'NumberOfFeatures'
y_axis_label = 'Number of Features'

type = 'sigmoid'
if type == 'rf':
    results_file = '/home/vanrijn/publications/AutoML2017/plot/data/rf_ranks.csv'
    colors = OrderedDict([('min. samples leaf', 'm'), ('max. features', 'b'), ('bootstrap', 'c'), ('split criterion', 'g'), ('min. samples split', 'y'), ('imputation', 'r')])
elif type == 'rbf':
    results_file = '/home/vanrijn/experiments/PIMP_flow6952_2017_08_19_15:06:53/ranks.csv'
    colors = {'tol':'m', 'strategy': 'c', 'shrinking': 'b', 'gamma': 'g', 'C': 'r'}
elif type == 'poly':
    results_file = '/home/vanrijn/experiments/PIMP_flow6952_2017_08_19_15:07:16/ranks.csv'
    colors =  {'degree': 'k', 'C': 'y', 'coef0': 'm', 'tol': 'c', 'strategy': 'b', 'gamma': 'g', 'shrinking': 'r'}
elif type == 'sigmoid':
    results_file = '/home/vanrijn/experiments/PIMP_flow6952_2017_08_19_15:07:25/ranks.csv'
    colors = {'shrinking': 'r', 'C': 'g', 'strategy': 'b', 'coef0': 'm', 'tol': 'k', 'gamma': 'c'}
else:
    results_file = '/home/vanrijn/publications/AutoML2017/plot/data/ada_ranks.csv'
    colors = OrderedDict([('max. depth', 'm'),  ('learning rate', 'b'), ('algorithm', 'g'), ('iterations', 'y'), ('imputation', 'r')])


x_vals = {}
y_vals = {}
area = {}
with open(results_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 0

    for row in reader:
        count += 1
        # if count >= 10: break;

        task_id = row['task_id'][5:]
        del row['task_id']

        best_param = max(row, key=lambda k: float(row[k]))
        value = row[best_param]
        print(task_id, best_param, value)

        if best_param not in x_vals:
            x_vals[best_param] = []
            y_vals[best_param] = []
            area[best_param] = []

        task = openml.tasks.get_task(task_id)
        x_vals[best_param].append(float(task.get_dataset().qualities[x_axis_feature]))
        y_vals[best_param].append(float(task.get_dataset().qualities[y_axis_feature]))
        area[best_param].append(float(value) * 50)

# maintans a list of the params that were at least once most important
all_params = set(x_vals.keys())

if len(all_params - colors.keys()) > 0:
    raise ValueError()

plotted_items = []
legend_keys = []
for param, value in colors.items():
    if param in all_params:
        occurances = len(x_vals[param])
        current = plt.scatter(x_vals[param], y_vals[param], s=area[param], c=colors[param], alpha=0.9)
        plotted_items.append(current)
        legend_keys.append(param)
    else:
        # param was never most important
        pass

legend = plt.legend(plotted_items, legend_keys, scatterpoints=1, loc='upper right')
for idx in range(len(plotted_items)):
    legend.legendHandles[idx]._sizes = [50]

plt.xscale("log")
plt.yscale("log")
plt.axis((450,100000,3,2100))

plt.xlabel(x_axis_label, fontsize='xx-large')
plt.ylabel(y_axis_label, fontsize='xx-large')
plt.savefig('result.pdf', bbox_inches='tight')