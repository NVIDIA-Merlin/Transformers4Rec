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

import logging
from abc import ABC

import numpy as np
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.data import Dataset, IterableDataset

from merlin_standard_lib import Registry, Schema, Tag

from ...utils import dependencies

logger = logging.getLogger(__name__)

dataloader_registry: Registry = Registry("torch.dataloader_loader")


class T4RecDataLoader(ABC):
    """
    Base Helper class to build dataloader from the schema with properties
    required by T4Rec Trainer class.
    """

    @classmethod
    def from_schema(
        self, schema: Schema, paths_or_dataset, batch_size, max_sequence_length, **kwargs
    ):
        # Build the data-loader from the schema
        raise NotImplementedError

    def set_dataset(self, paths_or_dataset):
        # set the dataset from paths
        # or from provided dataset
        raise NotImplementedError

    @classmethod
    def parse(cls, class_or_str):
        return dataloader_registry.parse(class_or_str)


if dependencies.is_pyarrow_available():
    import pyarrow.parquet as pq

    @dataloader_registry.register_with_multiple_names("pyarrow_builder", "pyarrow")
    class PyarrowDataLoader(T4RecDataLoader, PyTorchDataLoader):
        def __init__(
            self,
            paths_or_dataset,
            batch_size,
            max_sequence_length,
            cols_to_read=None,
            shuffle=False,
            shuffle_buffer_size=0,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            **kwargs,
        ):
            T4RecDataLoader.__init__(self)
            self.paths_or_dataset = paths_or_dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.shuffle_buffer_size = shuffle_buffer_size
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.max_sequence_length = max_sequence_length
            self.drop_last = drop_last

            self.set_dataset(cols_to_read=cols_to_read)

            PyTorchDataLoader.__init__(
                self,
                self.dataset,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            # set _batch_size attribute needed by HF trainer
            self._batch_size = self.batch_size

        def set_dataset(self, cols_to_read):
            """
            set the Parquet dataset

            Parameters
            ----------
            cols_to_read: str
                The list of features names to load
            """

            if isinstance(self.paths_or_dataset, ParquetDataset):
                dataset = self.paths_or_dataset
            dataset = ParquetDataset(
                self.paths_or_dataset,
                cols_to_read,
                seq_features_len_pad_trim=self.max_sequence_length,
            )
            if self.shuffle and self.shuffle_buffer_size > 0:
                dataset = ShuffleDataset(dataset, buffer_size=self.shuffle_buffer_size)

            self.dataset = dataset

        @classmethod
        def from_schema(
            cls,
            schema,
            paths_or_dataset,
            batch_size,
            max_sequence_length,
            continuous_features=None,
            categorical_features=None,
            targets=None,
            shuffle=False,
            shuffle_buffer_size=0,
            num_workers=1,
            pin_memory=True,
            **kwargs,
        ):
            """
            Instantiates ``PyarrowDataLoader`` from a ``DatasetSchema``.

            Parameters
            ----------
            schema: DatasetSchema
                Dataset schema
            paths_or_dataset: Union[str, Dataset]
                Path to paquet data of Dataset object.
            batch_size: int
                batch size of Dataloader.
            max_sequence_length: int
                The maximum length of list features.
            """

            categorical_features = (
                categorical_features or schema.select_by_tag(Tag.CATEGORICAL).column_names
            )
            continuous_features = (
                continuous_features or schema.select_by_tag(Tag.CONTINUOUS).column_names
            )
            targets = targets or schema.select_by_tag(Tag.TARGETS).column_names

            cols_to_read = categorical_features + continuous_features + targets

            return cls(
                paths_or_dataset,
                batch_size,
                max_sequence_length,
                cols_to_read=cols_to_read,
                shuffle=shuffle,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **kwargs,
            )


if dependencies.is_gpu_dataloader_available():
    from nvtabular.loader.torch import DLDataLoader
    from nvtabular.loader.torch import TorchAsyncItr as DataLoader

    from merlin_standard_lib.utils.misc_utils import validate_dataset

    class DLDataLoaderWrapper(DLDataLoader):
        """
        Setting the batch size directly to DLDataLoader makes it 3x slower.
        So we set as an alternative attribute and use it within
        T4Rec Trainer during evaluation
        # TODO : run experiments with new nvt dataloader
        """

        def __init__(self, *args, **kwargs) -> None:
            if "batch_size" in kwargs:
                self._batch_size = kwargs.pop("batch_size")
                super().__init__(*args, **kwargs)

    @dataloader_registry.register_with_multiple_names("nvtabular_dataloader", "nvtabular")
    class NVTabularDataLoader(T4RecDataLoader, DLDataLoaderWrapper):
        def __init__(
            self,
            paths_or_dataset,
            batch_size,
            max_sequence_length,
            conts=None,
            cats=None,
            labels=None,
            collate_fn=lambda x: x[0][0],
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
            sparse_as_dense=True,
            drop_last=False,
            schema=None,
            **kwargs,
        ):
            T4RecDataLoader.__init__(self)

            self.paths_or_dataset = paths_or_dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.max_sequence_length = max_sequence_length
            self.drop_last = drop_last

            self.set_dataset(buffer_size, engine, reader_kwargs)

            loader = DataLoader(
                self.dataset,
                cats,
                conts,
                labels,
                self.batch_size,
                shuffle,
                seed_fn=seed_fn,
                parts_per_chunk=parts_per_chunk,
                device=device,
                global_size=global_size,
                global_rank=global_rank,
                drop_last=drop_last,
                sparse_names=sparse_names,
                sparse_max=sparse_max,
                sparse_as_dense=sparse_as_dense,
            )

            DLDataLoaderWrapper.__init__(
                self,
                loader,
                collate_fn=collate_fn,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            self.schema = schema
            self.max_sequence_length = max_sequence_length

        def set_dataset(self, buffer_size, engine, reader_kwargs):
            dataset = validate_dataset(
                self.paths_or_dataset,
                self.batch_size,
                buffer_size,
                engine,
                reader_kwargs,
            )
            self.dataset = dataset

        @classmethod
        def from_schema(
            cls,
            schema: Schema,
            paths_or_dataset,
            batch_size,
            max_sequence_length,
            continuous_features=None,
            categorical_features=None,
            targets=None,
            collate_fn=lambda x: x[0][0],
            shuffle=True,
            buffer_size=0.06,
            parts_per_chunk=1,
            separate_labels=True,
            named_labels=False,
            sparse_names=None,
            sparse_max=None,
            **kwargs,
        ):
            """
               Instantitates ``NVTabularDataLoader`` from a ``DatasetSchema``.
            Parameters
            ----------
                schema: DatasetSchema
                    Dataset schema
                paths_or_dataset: Union[str, Dataset]
                    Path to paquet data of Dataset object.
                batch_size: int
                    batch size of Dataloader.
                max_sequence_length: int
                    The maximum length of list features.
            """
            categorical_features = (
                categorical_features or schema.select_by_tag(Tag.CATEGORICAL).column_names
            )
            continuous_features = (
                continuous_features or schema.select_by_tag(Tag.CONTINUOUS).column_names
            )
            targets = targets or schema.select_by_tag(Tag.TARGETS).column_names

            sparse_names = sparse_names or schema.select_by_tag(Tag.LIST).column_names
            sparse_max = sparse_max or {name: max_sequence_length for name in sparse_names}
            nvt_loader = cls(
                paths_or_dataset,
                batch_size=batch_size,
                max_sequence_length=max_sequence_length,
                labels=targets if separate_labels else [],
                cats=categorical_features if separate_labels else categorical_features + targets,
                conts=continuous_features,
                collate_fn=collate_fn,
                engine="parquet",
                shuffle=shuffle,
                buffer_size=buffer_size,  # how many batches to load at once
                parts_per_chunk=parts_per_chunk,
                sparse_names=sparse_names,
                sparse_max=sparse_max,
                schema=schema,
                **kwargs,
            )

            return nvt_loader


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
        # TODO define The shuffle method for pyarrow dataloader

    def __len__(self):
        return len(self.dataset)
