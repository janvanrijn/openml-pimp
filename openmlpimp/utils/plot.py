import math
import csv
import matplotlib
import Orange

def _critical_dist(numModels, numDatasets):
    # confidence values for alpha = 0.05. Index is the number of models (minimal two)
    alpha005 = [-1, -1, 1.959964233, 2.343700476, 2.569032073, 2.727774717,
                2.849705382, 2.948319908, 3.030878867, 3.10173026, 3.16368342,
                3.218653901, 3.268003591, 3.312738701, 3.353617959, 3.391230382,
                3.426041249, 3.458424619, 3.488684546, 3.517072762, 3.543799277,
                3.569040161, 3.592946027, 3.615646276, 3.637252631, 3.657860551,
                3.677556303, 3.696413427, 3.71449839, 3.731869175, 3.748578108,
                3.764671858, 3.780192852, 3.795178566, 3.809663649, 3.823679212,
                3.837254248, 3.850413505, 3.863181025, 3.875578729, 3.887627121,
                3.899344587, 3.910747391, 3.921852503, 3.932673359, 3.943224099,
                3.953518159, 3.963566147, 3.973379375, 3.98296845, 3.992343271,
                4.001512325, 4.010484803, 4.019267776, 4.02786973, 4.036297029,
                4.044556036, 4.05265453, 4.060596753, 4.068389777, 4.076037844,
                4.083547318, 4.090921028, 4.098166044, 4.105284488, 4.112282016,
                4.119161458, 4.125927056, 4.132582345, 4.139131568, 4.145576139,
                4.151921008, 4.158168297, 4.164320833, 4.170380738, 4.176352255,
                4.182236797, 4.188036487, 4.19375486, 4.199392622, 4.204952603,
                4.21043763, 4.215848411, 4.221187067, 4.22645572, 4.23165649,
                4.236790793, 4.241859334, 4.246864943, 4.251809034, 4.256692313,
                4.261516196, 4.266282802, 4.270992841, 4.275648432, 4.280249575,
                4.284798393, 4.289294885, 4.29374188, 4.298139377, 4.302488791]
    if numModels < 2:
        raise ValueError('needs at least 2 models')
    return alpha005[numModels] * math.sqrt((numModels * (numModels + 1)) / (6 * numDatasets))


def plot_nemenyi(algorithms, num_datasets, location):
    # and do the plotting
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    params = {'text.usetex' : True,
              'font.size' : 11,
              'font.family' : 'lmodern',
              'text.latex.unicode': True,
              }
    matplotlib.rcParams.update(params)

    # parameters controlling totalWidth and with of text (the latter is used both left and right)
    iWidth = 8
    iTextspace = 2

    algorithm_names = []
    algorithm_ranks = []
    for key, value in algorithms.items():
        algorithm_names.append(key.replace('_', '\_'))
        algorithm_ranks.append(value)

    cd = _critical_dist(len(algorithms), num_datasets)

    # print some statistics, for sanity checking
    print(algorithms)
    print("Number of Algorithms: %d" % len(algorithm_names))
    print("Number of Datasets  : %d" % num_datasets)
    print("Critical distance   : %f" % cd)

    # and plot
    Orange.evaluation.scoring.graph_ranks(algorithm_ranks,
                                          algorithm_names,
                                          cd=cd, filename=location,
                                          width=iWidth,
                                          textspace=iTextspace)


def to_csv_file(ranks_dict, location):
    hyperparameters = None
    for task_id, params in ranks_dict.items():
        hyperparameters = set(params)

    with open(location, 'w') as csvfile:
        fieldnames = ['task_id']
        fieldnames.extend(hyperparameters)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task_id, param_values in ranks_dict.items():
            result = {}
            result.update(param_values)
            result['task_id'] = 'Task %d' %task_id
            writer.writerow(result)
    pass


def to_csv_unpivot(ranks_dict, location):
    with open(location, 'w') as csvfile:
        fieldnames = ['task_id', 'param_id', 'param_name', 'variance_contribution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for task_id, param_values in ranks_dict.items():

            for param_name, variance_contribution in param_values.items():
                result = {'task_id' : task_id,
                          'param_id': param_name,
                          'param_name': param_name,
                          'variance_contribution': variance_contribution}
                writer.writerow(result)
    pass
