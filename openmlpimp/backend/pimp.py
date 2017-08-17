import os
import json
import tempfile
import openmlpimp

from pimp.importance.importance import Importance


class PimpBackend(object):

    @staticmethod
    def execute(save_folder, runhistory_location, configspace_location, modus='ablation', seed=1, reverse=True):
        # create scenario file

        with open(runhistory_location, 'r') as runhistory_filep:
            runhistory = json.load(runhistory_filep)

        scenario_dict = {'run_obj':'quality', 'deterministic': 1, 'paramfile': configspace_location}

        # TODO: tmp
        default_setup_id = runhistory['data'][0][0][0]

        trajectory_lines = openmlpimp.utils.runhistory_to_trajectory(runhistory, default_setup_id)
        if len(trajectory_lines) < 2:
            raise ValueError('trajectory file should contain at least two lines. ')

        traj_file = tempfile.NamedTemporaryFile('w', delete=False)
        for line in trajectory_lines:
            json.dump(line, traj_file)
            traj_file.write("\n")
        traj_file.close()

        importance = Importance(scenario_dict, runhistory_location,
                                traj_file=traj_file.name,
                                seed=seed,
                                save_folder=save_folder)

        try: os.makedirs(save_folder)
        except FileExistsError: pass

        for i in range(5):
            try:
                result = importance.evaluate_scenario(modus)
                filename = 'pimp_values_%s.json' %modus
                with open(os.path.join(save_folder, filename), 'w') as out_file:
                    json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
                importance.plot_results(name=os.path.join(save_folder, modus))
                return save_folder + "/" + filename
            except ZeroDivisionError as e:
                pass
        raise e