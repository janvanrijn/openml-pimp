import arff
import numpy as np
import os
from scipy.stats import wilcoxon

# does a statistical test on the intermediate results of

directory_A = os.path.expanduser('~') + '/nemo/experiments/priorbased_experiments/libsvm_svc/kernel_rbf/uniform__bestN_10__ignore_logscale_False__inverse_holdout_False__oob_strategy_ignore'
directory_B = os.path.expanduser('~') + '/nemo/experiments/priorbased_experiments/libsvm_svc/kernel_rbf/kde__bestN_10__ignore_logscale_False__inverse_holdout_False__oob_strategy_ignore'

task_ids = set(os.listdir(directory_A)).intersection(set(os.listdir(directory_A)))
print('Found %d task_ids' %len(task_ids))

for task_id in task_ids:
    traceAPath = os.path.join(directory_A, task_id, 'trace.arff')
    traceBPath = os.path.join(directory_B, task_id, 'trace.arff')

    if not (os.path.isfile(traceAPath) and os.path.isfile(traceBPath)):
        continue

    with open(traceAPath, 'r') as traceAPointer:
        traceA = arff.load(traceAPointer)
        attribute_index = {attribute[0]: index for index, attribute in enumerate(traceA['attributes'])}
        valuesA = np.array([float(obs[attribute_index['evaluation']]) for obs in traceA['data']])

    with open(traceBPath, 'r') as traceBPointer:
        traceB = arff.load(traceBPointer)
        attribute_index = {attribute[0]: index for index, attribute in enumerate(traceB['attributes'])}
        valuesB = np.array([float(obs[attribute_index['evaluation']]) for obs in traceB['data']])

    T, p = wilcoxon(valuesA, valuesB)
    medianA = np.median(valuesA)
    medianB = np.median(valuesB)
    res = 'uniform'
    if medianA < medianB:
        res = 'kde'

    print(task_id, res)
