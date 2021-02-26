import os
import sys
import glob
import time
import itertools
import subprocess
from typing import NamedTuple, Dict, Any

import torch


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
    paths = [[p for p in glob.glob(os.path.join(path, files_filter_pattern))] for path in data_paths]
    return list(itertools.chain.from_iterable(paths))


def get_label_feature_name(feature_map: Dict[str, Any]) -> str:
    """
        Analyses the feature map config and returns the name of the label feature (e.g. item_id)
    """
    label_feature_config = list([k for k,v in feature_map.items() if 'is_label' in v and v['is_label']])

    if len(label_feature_config) == 0:
        raise ValueError('One feature have be configured as label (is_label = True)')
    if len(label_feature_config) > 1:
        raise ValueError('Only one feature can be selected as label (is_label = True)')
    label_name = label_feature_config[0]
    return label_name


def get_timestamp_feature_name(feature_map: Dict[str, Any]) -> str:
    """
        Analyses the feature map config and returns the name of the label feature (e.g. item_id)
    """
    timestamp_feature_name = list([k for k,v in feature_map.items() if v['dtype'] == 'timestamp'])

    if len(timestamp_feature_name) == 0:
        raise ValueError('No feature have be configured as timestamp (dtype = "timestamp")')
    if len(timestamp_feature_name) > 1:
        raise ValueError('Only one feature can be configured as timestamp (dtype = "timestamp")')
    timestamp_fname = timestamp_feature_name[0]
    return timestamp_fname


def get_parquet_files_names(data_args, time_indices, is_train, eval_on_test_set=False):
    if type(time_indices) is not list:
        time_indices = [time_indices]

    time_window_folders = [os.path.join(data_args.data_path, str(time_index).zfill(data_args.time_window_folder_pad_digits)) for time_index in time_indices]
    if is_train:
        data_filename = 'train.parquet'
    else:
        if eval_on_test_set:
            data_filename = 'test.parquet'
        else:
            data_filename = 'valid.parquet'

    parquet_paths = [os.path.join(folder, data_filename) for folder in time_window_folders]

    ##If paths are folders, list the parquet file within the folders
    #parquet_paths = get_filenames(parquet_paths, files_filter_pattern="*.parquet"

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
            print(message, end='\n' if newline else '', file=self.file)
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
        self._log('{}Done in {:.3f}s'.format('' if self.one_line else self.message, time.time()-self.start))
