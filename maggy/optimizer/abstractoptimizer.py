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

from abc import abstractmethod

from maggy.core import controller


class AbstractOptimizer(controller.Controller):
    def __init__(self):
        self.searchspace = None
        self.num_trials = None
        self.trial_store = None
        self.final_store = None
        self.direction = None

    @abstractmethod
    def initialize(self):
        """
        A hook for the developer to initialize the optimizer.
        """
        pass

    @abstractmethod
    def get_next_trial(self, trial=None):
        """
        Return a `Trial` to be assigned to an executor, or `None` if there are
        no trials remaining in the experiment.

        :rtype: `Trial` or `None`
        """
        pass

    @abstractmethod
    def finalize(self, trials):
        """
        This method will be called before finishing the experiment. Developers
        can implement this method e.g. for cleanup or extra logging.
        """
        pass
