#
#   Copyright 2021 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import os
import json

import numpy as np
from hops import hdfs as hopshdfs
from hops.experiment_impl.util import experiment_utils

from maggy import tensorboard
from maggy.environment import base


class Environment(base.Environment):
    def __init__(self):
        pass

    def set_ml_id(self, app_id, run_id):
        os.environ["HOME"] = os.getcwd()
        os.environ["ML_ID"] = str(app_id) + "_" + str(run_id)
        return str(app_id) + "_" + str(run_id)

    def create_experiment_dir(self, app_id, run_id):
        # TODO: maintin own version
        experiment_utils._create_experiment_dir(app_id, run_id)

    def init_ml_tracking(self, app_id, run_id):
        tensorboard._register(experiment_utils._get_logdir(app_id, run_id))

    def get_logdir(self, app_id, run_id):
        return experiment_utils._get_logdir(app_id, run_id)

    def log_searchspace(self, app_id, run_id, searchspace):
        tensorboard._write_hparams_config(self.get_logdir(app_id, run_id), searchspace)

    def populate_experiment_json(
        self, name, exp_function, description, app_id, direction, optimization_key
    ):
        return experiment_utils._populate_experiment(
            name,
            exp_function,
            "MAGGY",
            None,
            description,
            app_id,
            direction,
            optimization_key,
        )

    def log_experiment_json(self, app_id, run_id, json, op="INIT"):
        return experiment_utils._attach_experiment_xattr(
            self.set_ml_id(app_id, run_id), json, op
        )

    def log_final_experiment_json(
        self,
        experiment_json,
        metric,
        app_id,
        run_id,
        state,
        duration,
        best_logdir,
        optimization_key,
    ):
        """Attaches the experiment outcome as xattr metadata to the app directory.
        """
        logdir = self.get_logdir(app_id, run_id)
        outputs = self._build_summary_json(logdir)

        if outputs:
            hopshdfs.dump(outputs, logdir + "/.summary.json")

        if best_logdir:
            experiment_json["bestDir"] = best_logdir[len(hopshdfs.project_path()) :]
        experiment_json["optimizationKey"] = optimization_key
        experiment_json["metric"] = metric
        experiment_json["state"] = state
        experiment_json["duration"] = duration
        exp_ml_id = app_id + "_" + str(run_id)
        experiment_utils._attach_experiment_xattr(
            exp_ml_id, experiment_json, "FULL_UPDATE"
        )

    def _build_summary_json(self, logdir):
        """Builds the summary json to be read by the experiments service.
        """
        combinations = []

        for trial in hopshdfs.ls(logdir):
            if hopshdfs.isdir(trial):
                return_file = trial + "/.outputs.json"
                hparams_file = trial + "/.hparams.json"
                if hopshdfs.exists(return_file) and hopshdfs.exists(hparams_file):
                    metric_arr = experiment_utils._convert_return_file_to_arr(
                        return_file
                    )
                    hparams_dict = self._load_hparams(hparams_file)
                    combinations.append(
                        {"parameters": hparams_dict, "outputs": metric_arr}
                    )

        return json.dumps(
            {"combinations": combinations}, default=self.json_default_numpy
        )

    def _load_hparams(self, hparams_file):
        """Loads the HParams configuration from a hparams file of a trial.
        """
        hparams_file_contents = hopshdfs.load(hparams_file)
        hparams = json.loads(hparams_file_contents)

        return hparams

    @staticmethod
    def json_default_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(
                "Object of type {0}: {1} is not JSON serializable".format(
                    type(obj), obj
                )
            )
