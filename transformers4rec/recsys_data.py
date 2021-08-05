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
"""
Set data-specific schema, vocab sizes, and feature extract function.
"""

import logging
import math
from random import randint

import numpy as np
import pandas as pd

# Pyarrow dependencies
import pyarrow.parquet as pq
import torch
from torch.distributed import get_world_size
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


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


def get_dataset_len(data_paths):
    return sum([pq.ParquetFile(d_path).metadata.num_rows for d_path in data_paths])


def fetch_data_loader(
    data_args,
    training_args,
    feature_map,
    data_paths,
    is_train_set,
    shuffle_dataloader=False,
):

    if type(data_paths) is not list:
        data_paths = [data_paths]

    batch_size = (
        training_args.per_device_train_batch_size
        if is_train_set
        else training_args.per_device_eval_batch_size
    )

    if data_args.data_loader_engine == "petastorm":
        # Petastorm data loader dependencies
        from petastorm import make_batch_reader
        from petastorm.pytorch import DataLoader as PetaStormDataLoader
        from petastorm.unischema import UnischemaField

        class DataLoaderWithLen(PetaStormDataLoader):
            def __init__(self, *args, **kwargs):
                if "len" not in kwargs:
                    self.len = 0
                else:
                    self.len = kwargs.pop("len")

                super(DataLoaderWithLen, self).__init__(*args, **kwargs)

            def __len__(self):
                return self.len

        def transform_petastorm_schema(feature_map):
            schema = []
            for cname, cinfo in feature_map.items():
                if cinfo["dtype"] in ["categorical", "int", "timestamp"]:
                    dtype = np.int64
                elif cinfo["dtype"] == "float":
                    dtype = np.float
                schema.append(UnischemaField(cname, dtype, (None,), None, True))
            return schema

        # eval_data_path = transform_petastorm_schema(feature_map)

        data_len = get_dataset_len(data_paths)

        data_paths = ["file://" + p for p in data_paths]

        loader = DataLoaderWithLen(
            make_batch_reader(
                data_paths,
                num_epochs=None,
                # TODO: Fix Petastorm dataloader and provide parquet_schema variable
                schema_fields=parquet_schema,
                reader_pool_type=data_args.petastorm_reader_pool_type,
                workers_count=data_args.workers_count,
            ),
            batch_size=batch_size,
            len=math.ceil(data_len / batch_size),
        )

    elif data_args.data_loader_engine == "pyarrow":
        cols_to_read = feature_map.keys()

        dataset = ParquetDataset(
            data_paths,
            cols_to_read,
            seq_features_len_pad_trim=data_args.session_seq_length_max,
        )
        if shuffle_dataloader and training_args.shuffle_buffer_size > 0:
            dataset = ShuffleDataset(dataset, buffer_size=training_args.shuffle_buffer_size)
        loader = PyTorchDataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=training_args.dataloader_drop_last,
            num_workers=data_args.workers_count,
            pin_memory=True,
        )

    elif data_args.data_loader_engine == "nvtabular":

        loader = get_nvtabular_dataloader(
            data_args,
            training_args,
            feature_map,
            data_paths,
            batch_size,
            shuffle_dataloader,
        )

    return loader


def get_nvtabular_dataloader(
    data_args,
    training_args,
    feature_map,
    data_paths,
    batch_size,
    shuffle_dataloader=False,
):
    # NVTabular dependencies
    from nvtabular import Dataset as NVTDataset
    from nvtabular.loader.torch import DLDataLoader
    from nvtabular.loader.torch import TorchAsyncItr as NVTDataLoader

    class DLDataLoaderWrapper(DLDataLoader):
        def __init__(self, *args, **kwargs) -> None:
            if "batch_size" in kwargs:
                # Setting the batch size directly to DLDataLoader makes it 3x slower. So we set as an alternative attribute and use it within RecSysTrainer during evaluation
                self._batch_size = kwargs.pop("batch_size")
            super().__init__(*args, **kwargs)

    def dataloader_collate(inputs):
        # Gets only the features dict
        inputs = inputs[0][0]
        return inputs

    categ_features = []
    continuous_features = []
    for fname, fprops in feature_map.items():
        if fprops["dtype"] in ["categorical", "timestamp"]:
            categ_features.append(fname)
        elif fprops["dtype"] in ["float", "long"]:
            continuous_features.append(fname)
        else:
            raise NotImplementedError(
                "The dtype {} is not currently supported.".format(fprops["dtype"])
            )

    sparse_features_max = {
        fname: feature_map[fname]["pad_trim_length"]
        if fname in feature_map and "pad_trim_length" in feature_map[fname]
        else data_args.session_seq_length_max
        for fname in categ_features + continuous_features
    }

    # device_key = "devices" if nvtabular.__version__ < "0.5.1" else "device"
    dataloader_device = 0 if training_args.local_rank == -1 else training_args.local_rank
    # If use part_size argument when doing multi-gpu distributed data parallel training,
    # note that dataset row_group memory size should be smaller than the partition size and
    # number of row groups per file should be >= number of gpus in use.
    if data_args.nvt_part_size:
        dataset = NVTDataset(data_paths, engine="parquet", part_size=data_args.nvt_part_size)
    else:
        dataset = NVTDataset(
            data_paths, engine="parquet", part_mem_fraction=data_args.nvt_part_mem_fraction
        )
    global_size = None
    global_rank = None
    # If using DistributedDataParallel, gets the global number of GPUs (world_size) and the GPU for this process (local_rank).
    # Each GPU will be assigned to one process and the data loader will read different chunks of data for each GPU
    if training_args.local_rank != -1:
        global_size = get_world_size()
        global_rank = training_args.local_rank

    loader = NVTDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle_dataloader,
        global_size=global_size,
        global_rank=global_rank,
        device=dataloader_device,
        cats=categ_features,
        conts=continuous_features,
        labels=[],
        sparse_names=categ_features + continuous_features,
        sparse_max=sparse_features_max,
        sparse_as_dense=True,
        drop_last=training_args.dataloader_drop_last,
    )

    dl_loader = DLDataLoaderWrapper(loader, collate_fn=dataloader_collate, batch_size=batch_size)

    return dl_loader


def get_items_sorted_freq(train_data_paths, item_id_feature_name):
    dataframes = []
    for parquet_file in train_data_paths:
        df = pd.read_parquet(parquet_file, columns=[item_id_feature_name])
        dataframes.append(df)

    concat_df = pd.concat(dataframes)
    # Returns a series indexed by item ids and sorted by the item frequency values
    items_sorted_freq_series = (
        concat_df.explode(item_id_feature_name)
        .groupby(item_id_feature_name)
        .size()
        .sort_values(ascending=True)
    )

    return items_sorted_freq_series


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
        except:
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
