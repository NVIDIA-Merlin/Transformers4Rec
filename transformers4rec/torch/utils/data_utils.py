import logging
from random import randint

import numpy as np

# Pyarrow dependencies
import pyarrow.parquet as pq
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.data import Dataset, IterableDataset

from transformers4rec.utils.schema import DatasetSchema
from transformers4rec.utils.tags import Tag

try:
    import nvtabular
    from nvtabular.loader.tensorflow import _validate_dataset
    from nvtabular.loader.torch import DLDataLoader

    from transformers4rec.torch.data import DataLoader

except ImportError:
    nvtabular = None

logger = logging.getLogger(__name__)


class DataLoaderBuilder:
    """
    Base Helper class to build NVTabular or PyArrow dataloader
    """

    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        max_sequence_length,
        dataloader_drop_last=True,
        shuffle=False,
    ):
        self.paths_or_dataset = paths_or_dataset
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.dataloader_drop_last = dataloader_drop_last
        self.shuffle = shuffle

    def build_from_schema(self, schema: DatasetSchema):
        # Build the data-loader from the schema
        raise NotImplementedError

    def get_dataset(self, paths_or_dataset):
        # Load the dataset from paths
        # Or return dataset object
        raise NotImplementedError


class PyarrowDataLoaderBuilder(DataLoaderBuilder):
    # TODO fix device mismatch error when calling eval after training
    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        max_sequence_length,
        shuffle_buffer_size=0,
        num_workers=1,
        pin_memory=True,
        **kwargs,
    ):

        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        DataLoaderBuilder.__init__(
            self, paths_or_dataset, batch_size, max_sequence_length, **kwargs
        )

    def get_dataset(self, cols_to_read):
        if isinstance(self.paths_or_dataset, ParquetDataset):
            return self.paths_or_dataset
        dataset = ParquetDataset(
            self.paths_or_dataset,
            cols_to_read,
            seq_features_len_pad_trim=self.max_sequence_length,
        )
        return dataset

    def build_from_schema(
        self,
        schema,
        continuous_features=None,
        categorical_features=None,
        targets=None,
    ):
        categorical_features = (
            categorical_features or schema.select_by_tag(Tag.CATEGORICAL).column_names
        )
        continuous_features = (
            continuous_features or schema.select_by_tag(Tag.CONTINUOUS).column_names
        )
        targets = targets or schema.select_by_tag(Tag.TARGETS).column_names

        cols_to_read = categorical_features + continuous_features + targets

        dataset = self.get_dataset(cols_to_read)

        if self.shuffle and self.shuffle_buffer_size > 0:
            dataset = ShuffleDataset(dataset, buffer_size=self.shuffle_buffer_size)

        loader = PyTorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=self.dataloader_drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        # set _batch_size attribute needed by HF trainer
        loader._batch_size = self.batch_size
        return loader


if nvtabular is not None:

    class NVTDataLoaderBuilder(DataLoaderBuilder):
        def __init__(
            self,
            paths_or_dataset,
            batch_size,
            max_sequence_length,
            engine=None,
            buffer_size=0.1,
            reader_kwargs=None,
            shuffle=False,
            seed_fn=None,
            parts_per_chunk=1,
            device=None,
            global_size=None,
            global_rank=None,
            sparse_names=None,
            sparse_max=None,
            sparse_as_dense=False,
            **kwargs,
        ):

            self.engine = engine
            self.buffer_size = buffer_size
            self.reader_kwargs = reader_kwargs
            self.shuffle = (shuffle,)
            self.seed_fn = (seed_fn,)
            self.parts_per_chunk = parts_per_chunk
            self.device = (device,)
            self.global_size = global_size
            self.global_rank = global_rank
            self.sparse_names = sparse_names
            self.sparse_max = sparse_max
            self.sparse_as_dense = sparse_as_dense

            DataLoaderBuilder.__init__(
                self, paths_or_dataset, batch_size, max_sequence_length, **kwargs
            )

        def get_dataset(self):
            dataset = _validate_dataset(
                self.paths_or_dataset,
                self.batch_size,
                self.buffer_size,
                self.engine,
                self.reader_kwargs,
            )
            return dataset

        def build_from_schema(
            self,
            schema,
            continuous_features=None,
            categorical_features=None,
            targets=None,
            collate_fn=lambda x: x[0][0],
        ):
            class DLDataLoaderWrapper(DLDataLoader):
                def __init__(self, *args, **kwargs) -> None:
                    if "batch_size" in kwargs:
                        # Setting the batch size directly to DLDataLoader makes it 3x slower.
                        # So we set as an alternative attribute and use
                        # it within T4Rec Trainer during evaluation
                        self._batch_size = kwargs.pop("batch_size")
                    super().__init__(*args, **kwargs)

            dataset = self.get_dataset()
            loader = DataLoader.from_schema(
                schema,
                dataset,
                batch_size=self.batch_size,
                sparse_names=self.sparse_names,
                sparse_max=self.sparse_max,
                sparse_as_dense=self.sparse_as_dense,
                drop_last=self.dataloader_drop_last,
            )

            return DLDataLoaderWrapper(
                loader,
                collate_fn=collate_fn,
                batch_size=self.batch_size,
                drop_last=self.dataloader_drop_last,
            )


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, cols_to_read, seq_features_len_pad_trim):
        self.cols_to_read = cols_to_read
        self.data = pq.ParquetDataset(parquet_file).read(columns=self.cols_to_read).to_pandas()
        self.seq_features_len_pad_trim = seq_features_len_pad_trim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        df = self.data.loc[index]
        return {col: self.pad_seq_column_if_needed(df[col]) for col in df.index}

    def pad_seq_column_if_needed(self, values):
        if type(values) is np.ndarray:
            values = values[: self.seq_features_len_pad_trim]
            if len(values) < self.seq_features_len_pad_trim:
                placeholder = np.zeros(self.seq_features_len_pad_trim, dtype=values.dtype)
                placeholder[: len(values)] = values
                values = placeholder
            if isinstance(values[0], np.floating) and values.dtype is not np.float32:
                values = values.astype(np.float32)
            if isinstance(values[0], np.integer) and values.dtype is not np.int64:
                values = values.astype(np.int64)
        return values


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):

        logger.info("[SHUFFLE] INITIALIZING BUFFER_SIZE: {}".format(self.buffer_size))

        raise StopIteration()

        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except GeneratorExit:
            self.buffer_size = len(shufbuf)

        try:
            logger.info(
                "[SHUFFLE] RETRIEVING FROM BUFFER AND REPLACING FROM ITERATOR: {}".format(
                    len(shufbuf)
                )
            )
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration as e:
                    logger.info("[SHUFFLE] StopIteration EXCEPTION: {}".format(e))
                    break

            logger.info("[SHUFFLE] STARTING TO RETRIEVE ONLY FROM BUFFER: {}".format(len(shufbuf)))

            while len(shufbuf) > 0:
                yield shufbuf.pop()

            logger.info("[SHUFFLE] FINISHED ITERATING")

        except GeneratorExit:
            pass

    def __len__(self):
        return len(self.dataset)
