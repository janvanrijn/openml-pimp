import matplotlib.pyplot as plt
import openml
import csv

x_axis_feature = 'NumberOfInstances'
x_axis_label = 'Number of Instances'

y_axis_feature = 'NumberOfFeatures'
y_axis_label = 'Number of Features'

results_file = '/home/vanrijn/publications/AutoML2017/plot/data/ada_ranks.csv'


x_vals = {}
y_vals = {}
area = {}
with open(results_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 0
    for row in reader:
        count+=1
        if count >= 50: break

        task_id = row['task_id'][5:]
        del row['task_id']

        best_param = max(row, key=lambda k: float(row[k]))
        value = row[best_param]
        print(task_id)

        if best_param not in x_vals:
            x_vals[best_param] = []
            y_vals[best_param] = []
            area[best_param] = []

        task = openml.tasks.get_task(task_id)
        x_vals[best_param].append(float(task.get_dataset().qualities[x_axis_feature]))
        y_vals[best_param].append(float(task.get_dataset().qualities[y_axis_feature]))
        area[best_param].append(float(value) * 50)

all_params = list(x_vals.keys())

colors = ['b', 'c', 'y', 'm', 'r']

plotted_items = []
for idx, param in enumerate(all_params):
    occurances = len(x_vals[param])
    current = plt.scatter(x_vals[param], y_vals[param], s=area[param], c=colors[idx], alpha=0.9)
    plotted_items.append(current)

plt.legend(plotted_items, all_params, scatterpoints=1, loc='lower left')

plt.xscale("log")
plt.yscale("log")
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.savefig('result.pdf', bbox_inches='tight')