import matplotlib.pyplot as plt
import openml
import csv

x_axis_feature = 'NumberOfInstances'
x_axis_label = 'Number of Data points'

y_axis_feature = 'NumberOfFeatures'
y_axis_label = 'Number of Features'

type = 'rf'
if type == 'rf':
    results_file = '/home/vanrijn/publications/AutoML2017/plot/data/rf_ranks.csv'
    colors = {'imputation': 'r', 'samples split': 'y', 'split criterion': 'g', 'bootstrap': 'c', 'max. features': 'b', 'min. samples leaf': 'm'}
else:
    results_file = '/home/vanrijn/publications/AutoML2017/plot/data/ada_ranks.csv'
    colors = {'imputation': 'r', 'iterations': 'y', 'algorithm': 'g', 'learning rate': 'b', 'max. depth': 'm'}


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

all_params = list(x_vals.keys())

plotted_items = []
legend = []
for idx, param in enumerate(all_params):
    occurances = len(x_vals[param])
    current = plt.scatter(x_vals[param], y_vals[param], s=area[param], c=colors[param], alpha=0.9)
    plotted_items.append(current)

legend = plt.legend(plotted_items, all_params, scatterpoints=1, loc='upper right')
for idx in range(len(plotted_items)):
    legend.legendHandles[idx]._sizes = [50]

plt.xscale("log")
plt.yscale("log")
plt.axis((450,100000,1,2100))

plt.xlabel(x_axis_label, fontsize='xx-large')
plt.ylabel(y_axis_label, fontsize='xx-large')
plt.savefig('result.pdf', bbox_inches='tight')