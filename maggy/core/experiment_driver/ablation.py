#
#   Copyright 2020 Logical Clocks AB
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

import json

from maggy import util
from maggy.earlystop import NoStoppingRule
from maggy.ablation.ablationstudy import AblationStudy
from maggy.ablation.ablator.loco import LOCO
from maggy.ablation.ablator import AbstractAblator
from maggy.core.experiment_driver import base


class Driver(base.Driver):
    def __init__(
        self,
        ablator,
        config,
        name,
        description,
        direction,
        num_executors,
        hb_interval,
        log_dir,
    ):
        super().__init__(
            name, description, direction, config, num_executors, hb_interval, log_dir
        )
        # set up an ablation study experiment
        self.earlystop_check = NoStoppingRule.earlystop_check

        if not isinstance(self.config, AblationStudy):
            raise Exception(
                "The experiment's configuration should be an instance of "
                "maggy.ablation.AblationStudy, "
                "but it is {0} (of type '{1}').".format(
                    str(self.config), type(self.config).__name__
                )
            )

        if isinstance(ablator, str):
            if ablator.lower() == "loco":
                self.controller = LOCO(self.config, self._final_store)
                self.num_trials = self.controller.get_number_of_trials()
                if self.num_executors > self.num_trials:
                    self.num_executors = self.num_trials
            else:
                raise Exception(
                    "The experiment's ablation study policy should either be a string ('loco') "
                    "or a custom policy that is an instance of maggy.ablation.ablation.AbstractAblator, "
                    "but it is {0} (of type '{1}').".format(
                        str(ablator), type(ablator).__name__
                    )
                )
        elif isinstance(ablator, AbstractAblator):
            self.controller = ablator
            print("Custom Ablator initialized. \n")
        else:
            raise Exception(
                "The experiment's ablation study policy should either be a string ('loco') "
                "or a custom policy that is an instance of maggy.ablation.ablation.AbstractAblator, "
                "but it is {0} (of type '{1}').".format(
                    str(ablator), type(ablator).__name__
                )
            )

        self.result = {"best_val": "n.a.", "num_trials": 0, "early_stopped": "n.a"}

        # Init controller
        self.controller.ablation_study = self.config
        self.controller.final_store = self._final_store
        self.controller.initialize()

    def log_string(self):
        log = (
            "Maggy Ablation "
            + str(self.result["num_trials"])
            + "/"
            + str(self.num_trials)
            + util._progress_bar(self.result["num_trials"], self.num_trials)
            + " - BEST Excludes"
            + json.dumps(self.result["best_config"])
            + " - metric "
            + str(self.result["best_val"])
        )
        return log
