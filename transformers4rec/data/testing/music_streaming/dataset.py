import pathlib

from transformers4rec.data.dataset import ParquetDataset

music_streaming_testing_data: ParquetDataset = ParquetDataset(pathlib.Path(__file__).parent)
