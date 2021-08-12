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
import contextlib
import os
from typing import Dict

import numpy as np
import tensorflow as tf
from nvtabular.io.dataset import Dataset
from nvtabular.loader.backend import DataLoader as BaseDataLoader
from nvtabular.loader.tf_utils import configure_tensorflow, get_dataset_schema_from_feature_columns
from nvtabular.ops import _get_embedding_order

from ..utils.schema import Schema
from ..utils.tags import Tag

from_dlpack = configure_tensorflow()


# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter,unexpected-keyword-arg,redundant-keyword-arg


def _validate_dataset(paths_or_dataset, batch_size, buffer_size, engine, reader_kwargs):
    # TODO: put this in parent class and allow
    # torch dataset to leverage as well?

    # if a dataset was passed, just return it
    if isinstance(paths_or_dataset, Dataset):
        return paths_or_dataset

    # otherwise initialize a dataset
    # from paths or glob pattern
    if isinstance(paths_or_dataset, str):
        files = tf.io.gfile.glob(paths_or_dataset)
        _is_empty_msg = "Couldn't find file pattern {} in directory {}".format(
            *os.path.split(paths_or_dataset)
        )
    else:
        # TODO: some checking around attribute
        # error here?
        files = list(paths_or_dataset)
        _is_empty_msg = "paths_or_dataset list must contain at least one filename"

    assert isinstance(files, list)
    if len(files) == 0:
        raise ValueError(_is_empty_msg)

    # implement buffer size logic
    # TODO: IMPORTANT
    # should we divide everything by 3 to account
    # for extra copies laying around due to asynchronicity?
    reader_kwargs = reader_kwargs or {}
    if buffer_size >= 1:
        if buffer_size < batch_size:
            reader_kwargs["batch_size"] = int(batch_size * buffer_size)
        else:
            reader_kwargs["batch_size"] = buffer_size
    else:
        reader_kwargs["part_mem_fraction"] = buffer_size
    return Dataset(files, engine=engine, **reader_kwargs)


def _validate_schema(feature_columns, cat_names, cont_names):
    _uses_feature_columns = feature_columns is not None
    _uses_explicit_schema = (cat_names is not None) or (cont_names is not None)
    if _uses_feature_columns and _uses_explicit_schema:
        raise ValueError(
            "Passed `feature_column`s and explicit column names, must be one or the other"
        )
    elif _uses_feature_columns:
        return get_dataset_schema_from_feature_columns(feature_columns)
    elif _uses_explicit_schema:
        cat_names = cat_names or []
        cont_names = cont_names or []
        return cat_names, cont_names
    else:
        raise ValueError(
            "Must either pass a list of TensorFlow `feature_column`s "
            "or explicit `cat_name` and `cont_name` column name lists."
        )


class DataLoader(tf.keras.utils.Sequence, BaseDataLoader):
    """
    Infinite generator used to asynchronously iterate through CSV or Parquet
    dataframes on GPU by leveraging an NVTabular `Dataset`. Applies preprocessing
    via NVTabular `Workflow` objects and outputs tabular dictionaries of TensorFlow
    Tensors via `dlpack <https://github.com/dmlc/dlpack>`_. Useful for training tabular models
    built in Keras and trained via
    `tf.keras.Model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_.

    The data loading scheme is implemented by loading, preprocessing, and
    batching data in an asynchronous thread. The amount of randomness in
    shuffling is controlled by the `buffer_size` and `parts_per_chunk`
    kwargs. At load time, sub-chunks of data with size controlled by
    `buffer_size` are loaded from random partitions in the dataset,
    and `parts_per_chunk` of them are concatenated into a single chunk,
    shuffled, and split into batches. This means that each chunk has
    `buffer_size*parts_per_chunk` rows, and due to the asynchronous
    nature of the dataloader that means there are, including the batch
    being processed by your network, `3*buffer_size*parts_per_chunk`
    rows of data in GPU memory at any given time. This means that
    for a fixed memory budget, using more `parts_per_chunk` will
    come at the expense of smaller `buffer_size`, increasing the number
    of reads and reducing throughput. The goal should be to maximize the
    total amount of memory utilized at once without going OOM and with
    the fewest number of reads to meet your epoch-level randomness needs.

    An important thing to note is that TensorFlow's default behavior
    is to claim all GPU memory for itself at initialziation time,
    which leaves none for NVTabular to load or preprocess data.
    As such, we attempt to configure TensorFlow to restrict
    its memory allocation on a given GPU using the environment variables
    `TF_MEMORY_ALLOCATION` and `TF_VISIBLE_DEVICE`. If `TF_MEMORY_ALLOCATION < 1`,
    it will be assumed that this refers to a fraction of free GPU
    memory on the given device. Otherwise, it will refer to an explicit
    allocation amount in MB. `TF_VISIBLE_DEVICE` should be an integer GPU
    index.

    Iterator output is of the form `(dict(features), list(labels))`,
    where each element of the features dict is a
    `feature_name: feature_tensor`  and each elemtn of the labels
    list is a tensor, and all tensors are of shape `(batch_size, 1)`.
    Note that this means vectorized continuous and multi-hot categorical
    features are not currently supported.
    The underlying NVTabular `Dataset` object is stored in the `data`
    attribute, and should be used for updating NVTabular `Workflow`
    statistics::

        workflow = nvt.Workflow(...)
        dataset = KerasSequenceLoader(...)
        workflow.update_stats(dataset.data.to_iter(), record_stats=True)

    Parameters
    -------------
    - paths_or_dataset: str or list(str)
        Either a string representing a file pattern (see `tf.glob` for
        pattern rules), a list of filenames to be iterated through, or
        a Dataset object, in which case `buffer_size`, `engine`, and
        `reader_kwargs` will be ignored
    - batch_size: int
        Number of samples to yield at each iteration
    - label_names: list(str)
        Column name of the target variable in the dataframe specified by
        `paths_or_dataset`
    - feature_columns: list(tf.feature_column) or None
        A list of TensorFlow feature columns representing the inputs
        exposed to the model to be trained. Columns with parent columns
        will climb the parent tree, and the names of the columns in the
        unique set of terminal columns will be used as the column names.
        If left as None, must specify `cat_names` and `cont_names`
    - cat_names: list(str) or None
        List of categorical column names. Ignored if `feature_columns` is
        specified
    - cont_names: list(str) or None
        List of continuous column names. Ignored if `feature_columns` is
        specified
    - engine: {'csv', 'parquet', None}, default None
        String specifying the type of read engine to use. If left as `None`,
        will try to infer the engine type from the file extension.
    - shuffle: bool, default True
        Whether to shuffle chunks of batches before iterating through them.
    - buffer_size: float or int
        If `0 <  buffer_size < 1`, `buffer_size` will refer to the fraction of
        total GPU memory to occupy with a buffered chunk. If `1 < buffer_size <
        batch_size`, the number of rows read for a buffered chunk will
        be equal to `int(buffer_size*batch_size)`. Otherwise, if `buffer_size >
        batch_size`, `buffer_size` rows will be read in each chunk (except for
        the last chunk in a dataset, which will, in general, be smaller).
        Larger chunk sizes will lead to more efficieny and randomness,
        but require more memory.
    - device: None
        Which GPU device to load from. Ignored for now
    - parts_per_chunk: int
        Number of dataset partitions with size dictated by `buffer_size`
        to load and concatenate asynchronously. More partitions leads to
        better epoch-level randomness but can negatively impact throughput
    - reader_kwargs: dict
        extra kwargs to pass when instantiating the underlying
        `nvtabular.Dataset`
    sparse_list : list(str) or None
        list with column names of columns that should be represented as sparse tensors
    sparse_max : dict
        dictionary of key: column_name + value: integer representing max sequence length for column
    sparse_dense : bool
        bool value to activate transforming sparse tensors to dense
    """

    _use_nnz = True

    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        label_names,
        feature_columns=None,
        cat_names=None,
        cont_names=None,
        engine=None,
        shuffle=True,
        seed_fn=None,
        buffer_size=0.1,
        device=None,
        parts_per_chunk=1,
        reader_kwargs=None,
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
        cat_names, cont_names = _validate_schema(feature_columns, cat_names, cont_names)

        # sort the ccolumns to avoid getting incorrect output
        # (https://github.com/NVIDIA/NVTabular/issues/412)
        cat_names = _get_embedding_order(cat_names)
        cont_names = _get_embedding_order(cont_names)

        device = device or 0
        BaseDataLoader.__init__(
            self,
            dataset,
            cat_names,
            cont_names,
            label_names,
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
        self._map_fns = []
        self.schema = schema

    @classmethod
    def from_directory(
        cls,
        directory,
        batch_size,
        shuffle=True,
        buffer_size=0.06,
        parts_per_chunk=1,
        separate_labels=True,
        named_labels=True,
        schema_path=None,
        continuous_features=None,
        categorical_features=None,
        targets=None,
    ):
        schema_path = schema_path or os.path.join(directory, "schema.pb")
        if not os.path.exists(schema_path):
            raise ValueError("Can't load from directory without a schema.")

        schema = Schema.from_schema(schema_path)

        categorical_features = (
            categorical_features or schema.select_by_tag(Tag.CATEGORICAL).column_names
        )
        continuous_features = (
            continuous_features or schema.select_by_tag(Tag.CONTINUOUS).column_names
        )
        targets = targets or schema.select_by_tag(Tag.TARGETS).column_names

        tf_dataset = cls(
            directory,
            batch_size=batch_size,
            label_names=targets if separate_labels else [],
            cat_names=categorical_features if separate_labels else categorical_features + targets,
            cont_names=continuous_features,
            engine="parquet",
            shuffle=shuffle,
            buffer_size=buffer_size,  # how many batches to load at once
            parts_per_chunk=parts_per_chunk,
            schema=schema,
        )

        if named_labels and separate_labels and len(targets) > 1:
            return tf_dataset.map(lambda X, y: (X, dict(zip(targets, y))))

        return tf_dataset

    def __len__(self):
        """
        recreating since otherwise Keras yells at you
        """
        # TODO: what's a better way to do this inheritance
        # of the appropriate methods? A Metaclass?
        BaseDataLoader.stop(self)
        return BaseDataLoader.__len__(self)

    def __getitem__(self, idx):
        """
        implemented exclusively for consistency
        with Keras model.fit. Does not leverage
        passed idx in any way
        """
        return BaseDataLoader.__next__(self)

    def map(self, fn):
        """
        Applying a function to each batch.

        This can for instance be used to add `sample_weight` to the model.
        """
        self._map_fns.append(fn)

        return self

    @property
    def output_shapes(self) -> Dict[str, tf.TensorShape]:
        sizes = {}

        for feature in self.schema.feature:
            name = feature.name
            if feature.HasField("value_count"):
                sizes[name] = tf.TensorShape([self.batch_size, feature.value_count.max])
            elif feature.HasField("shape"):
                sizes[name] = tf.TensorShape(
                    [self.batch_size] + [d.size for d in feature.shape.dim]
                )
            else:
                sizes[name] = tf.TensorShape([self.batch_size, 1])

        return sizes

    def head(self):
        return next(self)

    def head_feature_shapes(self):
        head = self.head()[0]

        shapes = {}

        for key, val in head.items():
            if isinstance(val, tuple):
                shapes[key] = (val[0].shape, val[1].shape)

            else:
                shapes[key] = val.shape

        return shapes

    @contextlib.contextmanager
    def _get_device_ctx(self, dev):
        # with tf.device("/device:GPU:{}".format(dev)) as tf_device:
        #     # tf.device changes the cupy cuda device, which breaks us on multigpu
        #     # force cupy to still use the device we expect
        #     cupy.cuda.Device(dev).use()
        #     yield tf_device
        # commenting out since device statements cause
        # RuntimeErrors when exiting if two dataloaders
        # are running at once (e.g. train and validation)
        if dev != "cpu":
            yield tf.device("/GPU:" + str(dev))
        else:
            # https://www.tensorflow.org/guide/gpu#manual_device_placement
            yield tf.device("/device:CPU:0")

    def _split_fn(self, tensor, idx, axis=0):
        return tf.split(tensor, idx, axis=axis)

    def _tensor_split(self, tensor, idx, axis=0):
        """
        Same function as above but need this method
        for api match.
        """
        return tf.split(tensor, idx, axis=axis)

    def create_keras_inputs(self, sparse_columns=None):
        if sparse_columns is None:
            sparse_columns = []
        inputs = {}

        for col in self.cont_names:
            inputs[col] = tf.keras.Input(name=col, dtype=tf.float32, shape=(None, 1))

        for col in self.cat_names:
            if col not in sparse_columns:
                inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(None, 1))
            else:
                inputs[col] = (
                    tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,)),
                    tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,)),
                )

        return inputs

    @property
    def _LONG_DTYPE(self):
        return tf.int64

    @property
    def _FLOAT32_DTYPE(self):
        return tf.float32

    def _pack(self, gdf):
        if isinstance(gdf, np.ndarray):
            return gdf
        if hasattr(gdf, "to_dlpack") and callable(getattr(gdf, "to_dlpack")):
            return gdf.to_dlpack()
        elif hasattr(gdf, "to_numpy") and callable(getattr(gdf, "to_numpy")):
            return gdf.to_numpy()
        return gdf.toDlpack()

    def _unpack(self, gdf):
        if hasattr(gdf, "shape"):
            return tf.convert_to_tensor(gdf)
        return from_dlpack(gdf)

    def _to_tensor(self, gdf, dtype=None):
        if gdf.empty:
            return

        # checks necessary because of this bug
        # https://github.com/tensorflow/tensorflow/issues/42660
        if len(gdf.shape) == 1 or gdf.shape[1] == 1:
            dlpack = self._pack(gdf)
        elif gdf.shape[0] == 1:
            dlpack = self._pack(gdf.values[0])
        else:
            dlpack = self._pack(gdf.values.T)

        # catch error caused by tf eager context
        # not being initialized
        try:
            x = self._unpack(dlpack)
        except AssertionError:
            tf.random.uniform((1,))
            x = self._unpack(dlpack)

        if gdf.shape[0] == 1:
            # batch size 1 so got squashed to a vector
            x = tf.expand_dims(x, 0)
        elif len(gdf.shape) == 1 or len(x.shape) == 1:
            # sort of a generic check for any other
            # len(shape)==1 case, could probably
            # be more specific
            x = tf.expand_dims(x, -1)
        elif gdf.shape[1] > 1:
            # matrix which means we had to transpose
            # for the bug above, so untranspose
            x = tf.transpose(x)
        return x

    def _pull_values_offsets(self, values_offset):
        """
        values_offset is either a tuple (values, offsets) or just values.
        Values is a tensor.
        This method is used to turn a tensor into its sparse representation
        """
        # pull_values_offsets, return values offsets diff_offsets
        diff_offsets = None
        if isinstance(values_offset, tuple):
            values = tf.reshape(values_offset[0], [-1])
            diff_offsets = tf.cast(tf.reshape(values_offset[1], [-1]), dtype=tf.int64)
            offsets = tf.math.cumsum(diff_offsets)
        else:
            values = tf.reshape(values_offset, [-1])
            offsets = tf.arange(tf.shape(values)[0], dtype=tf.int64)
            diff_offsets = offsets[1:] - offsets[:-1]
        num_rows = len(offsets)
        return values, offsets, diff_offsets, num_rows

    def _get_max_seq_len(self, diff_offsets):
        # get_max_seq_len, return int
        return int(tf.math.reduce_max(diff_offsets))

    def _get_indices(self, offsets, diff_offsets):
        # Building the indices to reconstruct the sparse tensors
        row_ids = tf.range(len(offsets), dtype=tf.int64)

        row_ids_repeated = tf.repeat(row_ids, diff_offsets)
        row_offset_repeated = tf.repeat(offsets, diff_offsets)
        col_ids = tf.range(len(row_offset_repeated), dtype=tf.int64) - row_offset_repeated
        indices = tf.concat(
            values=[tf.expand_dims(row_ids_repeated, -1), tf.expand_dims(col_ids, -1)], axis=1
        )
        return indices

    def _get_sparse_tensor(self, values, indices, num_rows, seq_limit):
        sparse_tensor = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=[num_rows, seq_limit]
        )
        return sparse_tensor

    def _build_sparse_tensor(self, values, offsets, diff_offsets, num_rows, seq_limit):
        ragged = tf.RaggedTensor.from_row_lengths(values=values, row_lengths=diff_offsets)
        tensor = tf.RaggedTensor.from_tensor(ragged.to_tensor(shape=[None, seq_limit])).to_sparse()
        if self.sparse_as_dense:
            tensor = tf.sparse.to_dense(tensor)
        return tensor

    def _handle_tensors(self, cats, conts, labels):
        to_return = super()._handle_tensors(cats, conts, labels)

        for map_fn in self._map_fns:
            to_return = map_fn(*to_return)

        return to_return


class DataLoaderValidator(tf.keras.callbacks.Callback):
    # TODO: document
    _supports_tf_logs = True

    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        for X, y_true in self.dataloader:
            y_pred = self.model(X)

            # TODO: how do we want to handle the multi-output case?
            for metric in self.model.metrics:
                metric.update_state(y_true, y_pred)

        set_logs = {}
        for metric in self.model.metrics:
            set_logs[f"val_{metric.name}"] = metric.result().numpy()
        logs.update(set_logs)
        print(set_logs)
        return logs
