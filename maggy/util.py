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

"""Utility helper module for maggy experiments.
"""
import math
import json

import numpy as np
from pyspark import TaskContext

from hops import hdfs as hopshdfs
from hops.experiment_impl.util import experiment_utils

from maggy import constants
from maggy.core import exceptions

DEBUG = True


def _log(msg):
    """
    Generic log function (in case logging is changed from stdout later)

    :param msg: The msg to log
    :type msg: str
    """
    if DEBUG:
        print(msg)


def num_executors(sc):
    """
    Get the number of executors configured for Jupyter

    :param sc: The SparkContext to take the executors from.
    :type sc: [SparkContext
    :return: Number of configured executors for Jupyter
    :rtype: int
    """
    try:
        return int(sc._conf.get("spark.dynamicAllocation.maxExecutors"))
    except:  # noqa: E722
        raise RuntimeError(
            "Failed to find spark.dynamicAllocation.maxExecutors property, \
            please select your mode as either Experiment, Parallel \
            Experiments or Distributed Training."
        )


def get_partition_attempt_id():
    """Returns partitionId and attemptNumber of the task context, when invoked
    on a spark executor.
    PartitionId is ID of the RDD partition that is computed by this task.
    The first task attempt will be assigned attemptNumber = 0, and subsequent
    attempts will have increasing attempt numbers.
    Returns:
        partitionId, attemptNumber -- [description]
    """
    task_context = TaskContext.get()
    return task_context.partitionId(), task_context.attemptNumber()


def _progress_bar(done, total):

    done_ratio = done / total
    progress = math.floor(done_ratio * 30)

    bar = "["

    for i in range(30):
        if i < progress:
            bar += "="
        elif i == progress:
            bar += ">"
        else:
            bar += "."

    bar += "]"
    return bar


def _handle_return_val(return_val, log_dir, optimization_key, log_file):
    """Handles the return value of the user defined training function.
    """
    experiment_utils._upload_file_output(return_val, log_dir)

    # Return type validation
    if not optimization_key:
        raise ValueError("Optimization key cannot be None.")
    if not return_val:
        raise exceptions.ReturnTypeError(optimization_key, return_val)
    if not isinstance(return_val, constants.USER_FCT.RETURN_TYPES):
        raise exceptions.ReturnTypeError(optimization_key, return_val)
    if isinstance(return_val, dict) and optimization_key not in return_val:
        raise KeyError(
            "Returned dictionary does not contain optimization key with the "
            "provided name: {}".format(optimization_key)
        )

    # validate that optimization metric is numeric
    if isinstance(return_val, dict):
        opt_val = return_val[optimization_key]
    else:
        opt_val = return_val
        return_val = {optimization_key: opt_val}

    if not isinstance(opt_val, constants.USER_FCT.NUMERIC_TYPES):
        raise exceptions.MetricTypeError(optimization_key, opt_val)

    # for key, value in return_val.items():
    #    return_val[key] = value if isinstance(value, str) else str(value)

    return_val["log"] = log_file.replace(hopshdfs.project_path(), "")

    return_file = log_dir + "/.outputs.json"
    hopshdfs.dump(json.dumps(return_val, default=json_default_numpy), return_file)

    metric_file = log_dir + "/.metric"
    hopshdfs.dump(json.dumps(opt_val, default=json_default_numpy), metric_file)

    return opt_val


def _clean_dir(clean_dir, keep=[]):
    """Deletes all files in a directory but keeps a few.
    """
    if not hopshdfs.isdir(clean_dir):
        raise ValueError(
            "{} is not a directory. Use `hops.hdfs.delete()` to delete single "
            "files.".format(clean_dir)
        )
    for path in hopshdfs.ls(clean_dir):
        if path not in keep:
            hopshdfs.delete(path, recursive=True)


def json_default_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(
            "Object of type {0}: {1} is not JSON serializable".format(type(obj), obj)
        )


def seconds_to_milliseconds(time):
    return int(round(time * 1000))
