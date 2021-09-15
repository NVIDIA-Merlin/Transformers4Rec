import pathlib

from ...dataset import ParquetDataset

tabular_testing_data: ParquetDataset = ParquetDataset(pathlib.Path(__file__).parent)
