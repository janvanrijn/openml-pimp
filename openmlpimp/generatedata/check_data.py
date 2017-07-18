import openml
import sklearn

flow_id = 6969
task_id = 31
interesting_params = {6969: ['classifier__min_samples_leaf', 'classifier__min_samples_split'],
                      6970: ['classifier__base_estimator__max_depth', 'classifier__n_estimators']}

evaluations = openml.evaluations.list_evaluations("predictive_accuracy", flow=[flow_id], task=[task_id])
task = openml.tasks.get_task(task_id)


for run_id in evaluations.keys():
    print('checking run %d; original score %f' %(run_id, evaluations[run_id].value))
    run_online = openml.runs.get_run(run_id)

    model_local = openml.setups.initialize_model(evaluations[run_id].setup_id)
    params = model_local.get_params()

    score_online = run_online.get_metric_score(sklearn.metrics.accuracy_score)
    print('ONLINE: %s; Accuracy: %0.4f' % (task.get_dataset().name, score_online.mean()))

    run = openml.runs.run_model_on_task(task, model_local)

    score_local = run.get_metric_score(sklearn.metrics.accuracy_score)  # prints accuracy score
    print('LOCAL : %s; Accuracy: %0.4f' % (task.get_dataset().name, score_local.mean()))
