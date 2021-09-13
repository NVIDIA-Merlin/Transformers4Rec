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
import inspect
import itertools
import logging
import os
import sys
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def filter_kwargs(kwargs, thing_with_kwargs, filter_positional_or_keyword=True):
    sig = inspect.signature(thing_with_kwargs)
    if filter_positional_or_keyword:
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
    else:
        filter_keys = [param.name for param in sig.parameters.values()]
    filtered_dict = {
        filter_key: kwargs[filter_key] for filter_key in filter_keys if filter_key in kwargs
    }
    return filtered_dict


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
    paths = [glob.glob(os.path.join(path, files_filter_pattern)) for path in data_paths]
    return list(itertools.chain.from_iterable(paths))


def get_label_feature_name(feature_map: Dict[str, Any]) -> str:
    """
    Analyses the feature map config and returns the name of the label feature (e.g. item_id)
    """
    label_feature_config = list(
        k for k, v in feature_map.items() if "is_label" in v and v["is_label"]
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
    timestamp_feature_name = list(k for k, v in feature_map.items() if v["dtype"] == "timestamp")

    if len(timestamp_feature_name) == 0:
        raise Exception('No feature have be configured as timestamp (dtype = "timestamp")')
    if len(timestamp_feature_name) > 1:
        raise Exception('Only one feature can be configured as timestamp (dtype = "timestamp")')

    timestamp_fname = timestamp_feature_name[0]
    return timestamp_fname


def get_parquet_files_names(data_args, time_indices, is_train, eval_on_test_set=False):
    if not isinstance(time_indices, list):
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

    parquet_paths = [os.path.join(folder, data_filename) for folder in time_window_folders]

    # If paths are folders, list the parquet file within the folders
    # parquet_paths = get_filenames(parquet_paths, files_filter_pattern="*.parquet"

    return parquet_paths


class Timing:
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
        # pylint: disable=broad-except
        if self.default_logger:
            print(message, end="\n" if newline else "", file=self.file)
            try:
                self.file.flush()
            except Exception:
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


def validate_dataset(paths_or_dataset, batch_size, buffer_size, engine, reader_kwargs):
    """
    Util function to load NVTabular Dataset from disk

    Parameters
    ----------
    paths_or_dataset: Union[nvtabular.Dataset, str]
        Path  to dataset to load of nvtabular Dataset,
        if Dataset, return the object.
    batch_size: int
        batch size for Dataloader.
    buffer_size: float
        parameter, which refers to the fraction of batches
        to load at once.
    engine: str
        parameter to specify the file format,
        possible values are: ["parquet", "csv", "csv-no-header"].
    reader_kwargs: dict
        Additional arguments of the specified reader.
    """
    try:
        from nvtabular.io.dataset import Dataset
    except ImportError:
        raise ValueError("NVTabular is necessary for this function, please install: " "nvtabular.")

    # TODO: put this in parent class and allow
    # torch dataset to leverage as well?

    # if a dataset was passed, just return it
    if isinstance(paths_or_dataset, Dataset):
        return paths_or_dataset

    # otherwise initialize a dataset
    # from paths or glob pattern
    if isinstance(paths_or_dataset, str):
        files = glob.glob(paths_or_dataset)
        _is_empty_msg = "Couldn't find file pattern {} in directory {}".format(
            *os.path.split(paths_or_dataset)
        )
    else:
        # TODO: some checking around attribute
        # error here?
        files = list(paths_or_dataset)
        _is_empty_msg = "paths_or_dataset list must contain at least one filename"

    assert isinstance(files, list)
    if len(files) == 0:
        raise ValueError(_is_empty_msg)

    # implement buffer size logic
    # TODO: IMPORTANT
    # should we divide everything by 3 to account
    # for extra copies laying around due to asynchronicity?
    reader_kwargs = reader_kwargs or {}
    if buffer_size >= 1:
        if buffer_size < batch_size:
            reader_kwargs["batch_size"] = int(batch_size * buffer_size)
        else:
            reader_kwargs["batch_size"] = buffer_size
    else:
        reader_kwargs["part_mem_fraction"] = buffer_size
    return Dataset(files, engine=engine, **reader_kwargs)
