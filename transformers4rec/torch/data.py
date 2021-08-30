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
import os
from typing import Dict

import pandas as pd
import torch
from nvtabular.loader.torch import TorchAsyncItr as BaseDataLoader

from transformers4rec.utils.misc_utils import _validate_dataset

from ..utils.schema import DatasetSchema
from ..utils.tags import Tag
from .utils.torch_utils import get_output_sizes_from_schema

# from nvtabular.loader.backend import DataLoader as BaseDataLoader


class IterDL(torch.utils.data.IterableDataset):
    def __init__(self, file_paths, batch_size=1, shuffle=False):
        super().__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for file_path in self.file_paths:
            pdf = pd.read_parquet(file_path)
            for start in range(0, pdf.shape[0], self.batch_size):
                df = pdf[start : start + self.batch_size]
                if self.shuffle:
                    df = df.sample(frac=1).reset_index(drop=True)
                yield df


class DataLoader(BaseDataLoader):
    """This class creates batches of tensor. Each batch size is specified by the user.
    The data input requires an NVTabular dataset. Handles spillover to ensure all
    batches are the specified size until the final batch.

    Parameters
    -----------
    dataset : NVTabular dataset
    cats : [str]
        the list of categorical columns in the dataset
    conts : [str]
        the list of continuous columns in the dataset
    labels : [str]
        the list of label columns in the dataset
    batch_size : int
        the size of each batch to supply to the model
    shuffle : bool
        enable/disable shuffling of dataset
    parts_per_chunk : int
        number of partitions from the iterator, an NVTabular Dataset, to concatenate into a "chunk"
    devices : [int]
        list representing all available GPU IDs
    sparse_list : [str]
        list with column names of columns that should be represented as sparse tensors
    sparse_max : {str: int}
        dictionary of key: column_name + value: integer representing max sequence length for column
    sparse_dense : bool
        bool value to activate transforming sparse tensors to dense
    """

    def __init__(
        self,
        paths_or_dataset,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        engine=None,
        buffer_size=0.1,
        reader_kwargs=None,
        shuffle=False,
        seed_fn=None,
        parts_per_chunk=1,
        device=None,
        global_size=None,
        global_rank=None,
        drop_last=False,
        sparse_names=None,
        sparse_max=None,
        sparse_as_dense=False,
        schema=None,
    ):
        dataset = _validate_dataset(
            paths_or_dataset, batch_size, buffer_size, engine, reader_kwargs
        )

        BaseDataLoader.__init__(
            self,
            dataset,
            cats,
            conts,
            labels,
            batch_size,
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
        self.schema = schema

    @classmethod
    def from_schema(
        cls,
        schema: DatasetSchema,
        paths_or_dataset,
        batch_size,
        shuffle=True,
        buffer_size=0.06,
        parts_per_chunk=1,
        separate_labels=True,
        named_labels=False,
        continuous_features=None,
        categorical_features=None,
        targets=None,
        **kwargs
    ):
        categorical_features = (
            categorical_features or schema.select_by_tag(Tag.CATEGORICAL).column_names
        )
        continuous_features = (
            continuous_features or schema.select_by_tag(Tag.CONTINUOUS).column_names
        )
        targets = targets or schema.select_by_tag(Tag.TARGETS).column_names

        torch_dataset = cls(
            paths_or_dataset,
            batch_size=batch_size,
            labels=targets if separate_labels else [],
            cats=categorical_features if separate_labels else categorical_features + targets,
            conts=continuous_features,
            engine="parquet",
            shuffle=shuffle,
            buffer_size=buffer_size,  # how many batches to load at once
            parts_per_chunk=parts_per_chunk,
            schema=schema,
            **kwargs
        )
        return torch_dataset

    @classmethod
    def from_directory(
        cls,
        directory,
        batch_size,
        shuffle=True,
        buffer_size=0.06,
        parts_per_chunk=1,
        separate_labels=True,
        named_labels=False,
        schema_path=None,
        continuous_features=None,
        categorical_features=None,
        targets=None,
        **kwargs
    ):
        schema_path = schema_path or os.path.join(directory, "schema.pb")
        if not os.path.exists(schema_path):
            raise ValueError("Can't load from directory without a schema.")

        schema = DatasetSchema.from_schema(schema_path)

        categorical_features = (
            categorical_features or schema.select_by_tag(Tag.CATEGORICAL).column_names
        )
        continuous_features = (
            continuous_features or schema.select_by_tag(Tag.CONTINUOUS).column_names
        )
        targets = targets or schema.select_by_tag(Tag.TARGETS).column_names

        torch_dataset = cls(
            directory,
            batch_size=batch_size,
            labels=targets if separate_labels else [],
            cats=categorical_features if separate_labels else categorical_features + targets,
            conts=continuous_features,
            engine="parquet",
            shuffle=shuffle,
            buffer_size=buffer_size,  # how many batches to load at once
            parts_per_chunk=parts_per_chunk,
            schema=schema,
            **kwargs
        )

        # if named_labels and separate_labels:
        #     return tf_dataset.map(lambda X, y: (X, dict(zip(targets, y))))

        return torch_dataset

    @property
    def output_sizes(self) -> Dict[str, torch.Size]:
        return get_output_sizes_from_schema(self.schema, self.batch_size)

    # Building the indices to reconstruct the sparse tensors

    def _get_sparse_tensor(self, values, indices, num_rows, seq_limit):
        sparse_tensor = torch.sparse_coo_tensor(
            indices.T, values, torch.Size([num_rows, seq_limit]), device=self.device
        )
        if self.sparse_as_dense:
            sparse_tensor = sparse_tensor.to_dense()
        return sparse_tensor


class DLDataLoader(torch.utils.data.DataLoader):
    """
    This class is an extension of the torch dataloader.
    It is required to support the FastAI framework.
    """

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataset)
