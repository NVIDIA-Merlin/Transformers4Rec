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
import warnings
from abc import ABC

import numpy as np
import torch
from merlin.dataloader.torch import Loader
from merlin.models.utils.misc_utils import validate_dataset
from merlin.models.utils.registry import Registry
from merlin.schema import Tags
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.data import Dataset, IterableDataset

from merlin_standard_lib import Schema
from transformers4rec.torch.utils.padding import pad_batch

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
            target_names=None,
            shuffle=False,
            shuffle_buffer_size=0,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            **kwargs,
        ):
            T4RecDataLoader.__init__(self)
            warnings.warn(
                "The `pyarrow` data loader is deprecated and should be replaced "
                "by `merlin_dataloader`",
                DeprecationWarning,
            )
            self.paths_or_dataset = paths_or_dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.shuffle_buffer_size = shuffle_buffer_size
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.max_sequence_length = max_sequence_length
            self.drop_last = drop_last

            self.set_dataset(cols_to_read=cols_to_read, target_names=target_names)

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

        def set_dataset(self, cols_to_read, target_names):
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
                target_names=target_names,
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
                categorical_features or schema.select_by_tag(Tags.CATEGORICAL).column_names
            )
            continuous_features = (
                continuous_features or schema.select_by_tag(Tags.CONTINUOUS).column_names
            )
            targets = targets or schema.select_by_tag(Tags.TARGET).column_names

            cols_to_read = categorical_features + continuous_features + targets

            return cls(
                paths_or_dataset,
                batch_size,
                max_sequence_length,
                cols_to_read=cols_to_read,
                target_names=targets,
                shuffle=shuffle,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **kwargs,
            )


class DLDataLoader(PyTorchDataLoader):
    """
    This class is an extension of the torch dataloader.
    It is required to support the FastAI framework.

    Setting the batch size directly to DLDataLoader makes it 3x slower.
    So we set as an alternative attribute and use it within
    T4Rec Trainer during evaluation
    # TODO : run experiments with new merlin-dataloader
    """

    def __init__(self, *args, **kwargs) -> None:
        if "batch_size" in kwargs:
            self._batch_size = kwargs.pop("batch_size")
            super().__init__(*args, **kwargs)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataset)


@dataloader_registry.register_with_multiple_names(
    "merlin_dataloader", "merlin", "nvtabular_dataloader", "nvtabular"
)
class MerlinDataLoader(T4RecDataLoader, DLDataLoader):
    """
    This class extends the [Merlin data loader]
    (https://github.com/NVIDIA-Merlin/dataloader/blob/main/merlin/dataloader/torch.py).
    The data input requires a merlin.io.Dataset or a path to the data files.
    It also sets the dataset's schema with the necessary properties to prepare the input
    list features as dense tensors (i.e. padded to the specified `max_sequence_length`).
    The dense representation is required by the Transformers4Rec input modules.

    Parameters
    ----------
    paths_or_dataset: Union[str, merlin.io.Dataset]
        The dataset to load.
    batch_size: int
        The size of each batch to supply to the model.
    max_sequence_length: int
        The maximum sequence length to use for padding list columns.
        By default, `0` is used as the padding index.
    cats : List[str], optional
        The list of categorical columns in the dataset.
        By default None.
    conts: List[str], optional
        The list of continuous columns in the dataset.
        By default None.
    labels : List[str], optional
        The list of label columns in the dataset.
        By default None.
    shuffle : bool, optional
        Enable/disable shuffling of dataset.
        By default False.
    parts_per_chunk : int, optional
        The number of partitions from the iterator, an Merlin Dataset,
        to concatenate into a "chunk". By default 1.
    device : int, optional
        The device id of the selected GPU
        By default None.
    sparse_names : [str], optional
        List with column names of columns that should be represented as sparse tensors.
        By default None.
    sparse_max : Dict[str, int], optional
        A dictionary of key: column_name + value: integer representing the max sequence
        length for a list column.
        By default None.
    sparse_as_dense : bool, optional
        Boolean value to activate transforming sparse tensors to dense ones.
        By default None.
    drop_last: bool, optional
        Whether or not to drop the last batch in an epoch. This is useful when you need to
        guarantee that each batch contains exactly `batch_size` rows - since the last batch
        will usually contain fewer rows.
    seed_fn: callable
        Function used to initialize random state
    parts_per_chunk: int
        Number of dataset partitions with size dictated by `buffer_size`
        to load and concatenate asynchronously. More partitions leads to
        better epoch-level randomness but can negatively impact throughput
    global_size: int, optional
        When doing distributed training, this indicates the number of total processes that are
        training the model.
    global_rank: int, optional
        When doing distributed training, this indicates the local rank for the current process.
    schema: Schema, optional
         The `Schema` with the input features.
    reader_kwargs:
        Extra arguments to pass to the merlin.io.Dataset object, when the path to data files
        is provided in `paths_or_dataset` argument.
    row_groups_per_part: bool, optional
        If true, preserve the group partitions when loading the dataset from parquet files.
    collate_fn: Callable, optional
        A processing function to collect and prepare the list samples
        (tuple of (input, target) Tensor(s)) returned by the Merlin DataLoader.
    """

    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        max_sequence_length,
        conts=None,
        cats=None,
        labels=None,
        collate_fn=lambda x: x[0],
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
        row_groups_per_part=True,
        **kwargs,
    ):
        T4RecDataLoader.__init__(self)

        self.paths_or_dataset = paths_or_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_sequence_length = max_sequence_length
        self.drop_last = drop_last

        reader_kwargs = reader_kwargs or {}
        reader_kwargs["row_groups_per_part"] = row_groups_per_part
        self.set_dataset(buffer_size, engine, reader_kwargs)

        if (global_rank is not None) and (self.dataset.npartitions < global_size):
            logger.warning(
                "UserWarning: User is advised to repartition the parquet file before training "
                "so npartitions>=global_size. Cudf or pandas can be used for repartitioning "
                "eg. pdf.to_parquet('file.parquet',row_group_size=N_ROWS/NPARTITIONS) for pandas "
                "or gdf.to_parquet('file.parquet',row_group_size_rows=N_ROWS/NPARTITIONS) for cudf "
                "so that npartitions=nr_rows/row_group_size. Also ensure npartitions is divisible "
                "by number of GPUs to be used (eg. 2 or 4 partitions, if 2 GPUs will be used)."
            )
            self.dataset = self.dataset.repartition(npartitions=global_size)

        if (global_rank is not None) and (self.dataset.npartitions % global_size != 0):
            logger.warning(
                f"UserWarning: User is advised to set the number of partitions"
                f" ({self.dataset.npartitions}) divisible by the number of available"
                f" GPUs ({global_size}). This will divide the work equally among GPUs"
                " for DDP training and ensure optimal performance."
            )

        self.dataset.schema = self._augment_schema(
            self.dataset.schema, cats=cats, conts=conts, labels=labels
        )

        loader = Loader(
            self.dataset,
            self.batch_size,
            shuffle,
            seed_fn=seed_fn,
            parts_per_chunk=parts_per_chunk,
            device=device,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
        ).map(self._get_pad_fn(sparse_max))

        DLDataLoader.__init__(
            self,
            loader,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        self.schema = schema
        self.max_sequence_length = max_sequence_length

    @staticmethod
    def _get_pad_fn(padding_lengths):
        def pad_fn(x, y):
            new_x = pad_batch(x, padding_lengths)
            if y is not None and isinstance(y, dict):
                new_y = pad_batch(y, padding_lengths)
            else:
                new_y = y
            return new_x, new_y

        return pad_fn

    @staticmethod
    def _augment_schema(
        schema,
        cats=None,
        conts=None,
        labels=None,
    ):
        cats = cats or []
        conts = conts or []
        labels = labels or []

        schema = schema.select_by_name(conts + cats + labels)

        labels = [labels] if isinstance(labels, str) else labels
        for label in labels:
            schema[label] = schema[label].with_tags(Tags.TARGET)
        for label in cats:
            schema[label] = schema[label].with_tags(Tags.CATEGORICAL)
        for label in conts:
            schema[label] = schema[label].with_tags(Tags.CONTINUOUS)

        return schema

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
        collate_fn=lambda x: x[0],
        shuffle=True,
        buffer_size=0.06,
        parts_per_chunk=1,
        sparse_names=None,
        sparse_max=None,
        **kwargs,
    ):
        """
            Instantitates `MerlinDataLoader` from a ``DatasetSchema``.
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
            categorical_features or schema.select_by_tag(Tags.CATEGORICAL).column_names
        )
        continuous_features = (
            continuous_features or schema.select_by_tag(Tags.CONTINUOUS).column_names
        )
        targets = targets or schema.select_by_tag(Tags.TARGET).column_names
        schema = schema.select_by_name(categorical_features + continuous_features + targets)
        sparse_names = sparse_names or schema.select_by_tag(Tags.LIST).column_names
        sparse_max = sparse_max or {name: max_sequence_length for name in sparse_names}
        loader = cls(
            paths_or_dataset,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            labels=targets,
            cats=categorical_features,
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

        return loader


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, cols_to_read, target_names, seq_features_len_pad_trim):
        self.cols_to_read = cols_to_read
        self.target_names = target_names
        self.data = pq.ParquetDataset(parquet_file).read(columns=self.cols_to_read).to_pandas()
        self.seq_features_len_pad_trim = seq_features_len_pad_trim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        df = self.data.loc[index]
        input_features = list(set(self.cols_to_read).difference(self.target_names))
        inputs = {col: self.pad_seq_column_if_needed(df[col]) for col in input_features}
        targets = {col: self.pad_seq_column_if_needed(df[col]) for col in self.target_names}
        return inputs, targets

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
