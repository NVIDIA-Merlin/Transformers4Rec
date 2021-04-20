#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import glob
import itertools
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, NamedTuple

import torch

logger = logging.getLogger(__name__)


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def get_filenames(data_paths, files_filter_pattern="*"):
    paths = [
        [p for p in glob.glob(os.path.join(path, files_filter_pattern))]
        for path in data_paths
    ]
    return list(itertools.chain.from_iterable(paths))


def get_label_feature_name(feature_map: Dict[str, Any]) -> str:
    """
        Analyses the feature map config and returns the name of the label feature (e.g. item_id)
    """
    label_feature_config = list(
        [k for k, v in feature_map.items() if "is_label" in v and v["is_label"]]
    )

    if len(label_feature_config) == 0:
        raise ValueError("One feature have be configured as label (is_label = True)")
    if len(label_feature_config) > 1:
        raise ValueError("Only one feature can be selected as label (is_label = True)")
    label_name = label_feature_config[0]
    return label_name


def get_timestamp_feature_name(feature_map: Dict[str, Any]) -> str:
    """
        Analyses the feature map config and returns the name of the label feature (e.g. item_id)
    """
    timestamp_feature_name = list(
        [k for k, v in feature_map.items() if v["dtype"] == "timestamp"]
    )

    if len(timestamp_feature_name) == 0:
        raise Exception(
            'No feature have be configured as timestamp (dtype = "timestamp")'
        )
    if len(timestamp_feature_name) > 1:
        raise Exception(
            'Only one feature can be configured as timestamp (dtype = "timestamp")'
        )

    timestamp_fname = timestamp_feature_name[0]
    return timestamp_fname


def get_parquet_files_names(data_args, time_indices, is_train, eval_on_test_set=False):
    if type(time_indices) is not list:
        time_indices = [time_indices]

    time_window_folders = [
        os.path.join(
            data_args.data_path,
            str(time_index).zfill(data_args.time_window_folder_pad_digits),
        )
        for time_index in time_indices
    ]
    if is_train:
        data_filename = "train.parquet"
    else:
        if eval_on_test_set:
            data_filename = "test.parquet"
        else:
            data_filename = "valid.parquet"

    parquet_paths = [
        os.path.join(folder, data_filename) for folder in time_window_folders
    ]

    ##If paths are folders, list the parquet file within the folders
    # parquet_paths = get_filenames(parquet_paths, files_filter_pattern="*.parquet"

    return parquet_paths


class Timing(object):
    """A context manager that prints the execution time of the block it manages"""

    def __init__(self, message, file=sys.stdout, logger=None, one_line=True):
        self.message = message
        if logger is not None:
            self.default_logger = False
            self.one_line = False
            self.logger = logger
        else:
            self.default_logger = True
            self.one_line = one_line
            self.logger = None
        self.file = file

    def _log(self, message, newline=True):
        if self.default_logger:
            print(message, end="\n" if newline else "", file=self.file)
            try:
                self.file.flush()
            except:
                pass
        else:
            self.logger.info(message)

    def __enter__(self):
        self.start = time.time()
        self._log(self.message, not self.one_line)

    def __exit__(self, exc_type, exc_value, traceback):
        self._log(
            "{}Done in {:.3f}s".format(
                "" if self.one_line else self.message, time.time() - self.start
            )
        )


def get_object_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_object_size(v, seen) for v in obj.values()])
        size += sum([get_object_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_object_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_object_size(i, seen) for i in obj])
    return size
