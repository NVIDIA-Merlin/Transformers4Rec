import glob
import itertools
from subprocess import check_output


def get_filenames(data_paths):
    paths = [['file://' + p for p in glob.glob(path + "/*.parquet")] for path in data_paths]
    return list(itertools.chain.from_iterable(paths))


def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])


def get_dataset_len(data_paths):
    return sum(wc(f.replace('file://', '')) for f in data_paths)
