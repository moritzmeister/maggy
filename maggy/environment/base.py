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
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self):
        pass

    def set_ml_id(self, app_id, run_id):
        return str(app_id) + "_" + str(run_id)

    def validate_ml_id(self, app_id, run_id):
        """Validates if there was an experiment run previously from the same app id
        but from a different experiment (e.g. hops-util-py vs. maggy) module.
        """
        try:
            prev_ml_id = os.environ["ML_ID"]
        except KeyError:
            return app_id, (run_id + 1)
        prev_app_id, _, prev_run_id = prev_ml_id.rpartition("_")
        if prev_run_id == prev_ml_id:
            # means there was no underscore found in string
            raise ValueError(
                "Found a previous ML_ID with wrong format: {}".format(prev_ml_id)
            )
        if prev_app_id == app_id and int(prev_run_id) >= run_id:
            return app_id, (int(prev_run_id) + 1)
        return app_id, run_id

    @abstractmethod
    def create_experiment_dir(self, app_id, run_id):
        pass

    @abstractmethod
    def init_ml_tracking(self, app_id, run_id):
        pass

    @abstractmethod
    def get_logdir(self, app_id, run_id):
        return None

    @abstractmethod
    def log_searchspace(self, app_id, run_id):
        pass

    @abstractmethod
    def populate_experiment_json(
        self, name, exp_function, description, app_id, direction, optimization_key
    ):
        return {}

    @abstractmethod
    def log_experiment_json(self, app_id, run_id, json, op="INIT"):
        return {}

    @abstractmethod
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
        pass
