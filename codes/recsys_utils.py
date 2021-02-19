import os
import sys
import glob
import time
import itertools
import subprocess
from typing import NamedTuple

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
