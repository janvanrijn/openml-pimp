import arff
import openml
import openmlpimp
import sklearn
from time import gmtime, strftime


def get_time():
    return strftime("[%Y-%m-%d %H:%M:%S]", gmtime())


def fixed_parameters_to_suffix(fixed_parameters):
    if fixed_parameters is not None and len(fixed_parameters) > 0:
        save_folder_suffix = [param + '_' + str(fixed_parameters[param]) for param in sorted(fixed_parameters)]
        save_folder_suffix = '/' + '__'.join(save_folder_suffix)
    else:
        save_folder_suffix = '/vanilla'
    return save_folder_suffix


def do_run(task, optimizer, output_dir, internet_access=True):
    if internet_access:
        run = openml.runs.run_model_on_task(task, optimizer)
        score = run.get_metric_fn(sklearn.metrics.accuracy_score)

        print('%s [SCORE] Data: %s; Accuracy: %0.2f' % (openmlpimp.utils.get_time(), task.get_dataset().name, score.mean()))

        trace_arff = arff.dumps(run._generate_trace_arff_dict())
        run_xml = run._create_description_xml()
        predictions_arff = arff.dumps(run._generate_arff_dict())

        with open(output_dir + '/trace.arff', 'w') as f:
            f.write(trace_arff)
        with open(output_dir + '/run.xml', 'w') as f:
            f.write(run_xml)
        with open(output_dir + '/predictions.arff', 'w') as f:
            f.write(predictions_arff)
    else:
        res = openml.runs.functions._run_task_get_arffcontent(optimizer, task, task.class_labels)
        run = openml.runs.OpenMLRun(task_id=task.task_id, dataset_id=None, flow_id=None, model=optimizer)
        run.data_content, run.trace_content, run.trace_attributes, run.fold_evaluations, _ = res
        score = run.get_metric_fn(sklearn.metrics.accuracy_score)

        print('%s [SCORE] Data: %s; Accuracy: %0.2f' % (
        openmlpimp.utils.get_time(), task.get_dataset().name, score.mean()))

        trace_arff = arff.dumps(run._generate_trace_arff_dict())
        predictions_arff = arff.dumps(run._generate_arff_dict())

        with open(output_dir + '/trace.arff', 'w') as f:
            f.write(trace_arff)
        with open(output_dir + '/predictions.arff', 'w') as f:
            f.write(predictions_arff)
