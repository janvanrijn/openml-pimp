import matplotlib.pyplot as plt
import openml
import csv

from collections import OrderedDict

x_axis_feature = 'NumberOfInstances'
x_axis_label = 'Number of Data Points'

y_axis_feature = 'NumberOfFeatures'
y_axis_label = 'Number of Features'

type = 'rbf'
if type == 'rf':
    results_file = '/home/vanrijn/experiments/archive/20170908/rf/ranks.csv'
    colors = OrderedDict([('min. samples leaf', 'm'), ('max. features', 'b'), ('bootstrap', 'c'), ('criterion', 'g'), ('min. samples split', 'y'), ('strategy', 'r')])
elif type == 'rbf':
    results_file = '/home/vanrijn/experiments/archive/20170908/rbf/ranks.csv'
    colors = OrderedDict([('gamma', 'm'), ('complexity', 'b'), ('tolerance', 'g'), ('strategy', 'y'), ('shrinking', 'r')])
# elif type == 'poly':
#     results_file = '/home/vanrijn/experiments/archive/20170904/poly/ranks.csv'
#     colors = OrderedDict([('gamma', 'm'), ('coef0', 'mediumpurple'), ('complexity', 'c'), ('degree', 'green'), ('tolerance', 'limegreen'), ('imputation', 'y'), ('shrinking', 'r')])
elif type == 'sigmoid':
    results_file = '/home/vanrijn/experiments/archive/20170908/sigmoid/ranks.csv'
    colors = OrderedDict([('gamma', 'm'), ('complexity', 'b'),  ('coef0', 'c'), ('tolerance', 'g'), ('strategy', 'y'), ('shrinking', 'r')])
else:
    results_file = '/home/vanrijn/experiments/archive/20170908/adaboost/ranks.csv'
    colors = OrderedDict([('max. depth', 'm'),  ('learning rate', 'b'), ('algorithm', 'g'), ('iterations', 'y'), ('strategy', 'r')])

print(type, results_file)
print(colors)
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

undefined = all_params - colors.keys()
if len(undefined) > 0:
    raise ValueError('undefined parameters:', undefined)

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
plt.savefig('result_%s.pdf' %type, bbox_inches='tight')