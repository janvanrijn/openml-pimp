import csv


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
