import sys
import glob
import time
import itertools
import subprocess
from typing import NamedTuple

import torch


def get_filenames(data_paths):
    paths = [['file://' + p for p in glob.glob(path + "/*.parquet")] for path in data_paths]
    return list(itertools.chain.from_iterable(paths))


def wc(filename):
    try:
        num_lines = int(subprocess.check_output(["wc", "-l", filename], stderr=subprocess.STDOUT).split()[0])
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    return num_lines


def get_dataset_len(data_paths):
    return sum(wc(f.replace('file://', '')) for f in data_paths)


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
