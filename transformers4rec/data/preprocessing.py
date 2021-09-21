# type: ignore

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
import os
import shutil
import tempfile
from typing import TypeVar

from tqdm import tqdm

from merlin_standard_lib import Schema

LOG = logging.getLogger("transformers4rec")
FIRST_SEEN_ITEM_COL_NAME = "item_ts_first"

DataFrameType = TypeVar("DataFrameType")


def remove_consecutive_interactions(
    df: DataFrameType, session_id_col="session_id", item_id_col="item_id", timestamp_col="timestamp"
) -> DataFrameType:
    LOG.info("Count with in-session repeated interactions: {}".format(len(df)))
    # Sorts the dataframe by session and timestamp, to remove consecutive repetitions
    df = df.sort_values([session_id_col, timestamp_col])

    # Keeping only no consecutive repeated in session interactions
    session_is_last_session = df[session_id_col] == df[session_id_col].shift(1)
    item_is_last_item = df[item_id_col] == df[item_id_col].shift(1)
    df = df[~(session_is_last_session & item_is_last_item)]
    LOG.info("Count after removed in-session repeated interactions: {}".format(len(df)))

    return df


def add_item_first_seen_col_to_df(
    df: DataFrameType,
    item_id_column="item_id",
    timestamp_column="timestamp",
    first_seen_column_name=FIRST_SEEN_ITEM_COL_NAME,
) -> DataFrameType:
    items_first_ts_df = (
        df.groupby(item_id_column)
        .agg({timestamp_column: "min"})
        .reset_index()
        .rename(columns={timestamp_column: first_seen_column_name})
    )
    merged_df = df.merge(items_first_ts_df, on=[item_id_column], how="left")

    return merged_df


def session_aggregator(
    schema: Schema,
    data: DataFrameType,
    maximum_length: int = 20,
    minimum_length: int = 2,
    device: str = "gpu",
):
    """
    Util function to aggregate item interactions dataset at session level using NVTabular.
    It supports `cpu` and `gpu`.

    Parameters:
    ----------
    schema: Schema
        The schema objects describing the columns of data.
    data: Union[pandas.DataFrame, cudf.DataFrame]
        The data with row item interactions.
    maximum_length: int
        Trim all sessions to a maximum length.
    minimum_length: int
        Filter out sessions shorter than minimum_length.
    device: str
        Aggregate data using `cpu` or gpu `NVTabular` workflow

    Returns:
    -------
    session_data: Union[pandas.DataFrame, cudf.DataFrame]
        session-level dataset with list features.
    """
    try:
        import nvtabular as nvt
    except ImportError:
        raise ValueError("NVTabular is necessary for this function, please install it")

    if device == "cpu":
        import dask.dataframe as dd
        import pandas as pd

        data = dd.from_pandas(
            data if isinstance(data, pd.DataFrame) else data.to_pandas(), npartitions=3
        )
    else:
        try:
            import dask_cudf
        except ImportError:
            raise ValueError(
                "Rapids is necessary for running function in gpu, please install: " "dask_cudf."
            )
        data = dask_cudf.from_cudf(data, npartitions=3)

    # get item and session features
    item_features = schema.select_by_tag(["item"]).feature
    session_feat = schema.select_by_tag(["session"]).column_names
    groupby_dict = {k.name: ["list"] for k in item_features}
    for col in session_feat:
        groupby_dict[col] = ["first"]

    # retrieve session_id column
    session_column = schema.select_by_tag(["session_id"]).feature
    if not session_column:
        raise ValueError("Please provide a schema with `session_id` tagged feature")
    session_column = session_column[0]

    # define groupby operator
    groupby_feats = nvt.ColumnSelector(schema.column_names)
    groupby_features = groupby_feats >> nvt.ops.Groupby(
        groupby_cols=[session_column.name], aggs=groupby_dict, name_sep="-"
    )

    # max_lengths
    list_variables = [feat.name + "-list" for feat in item_features]
    groupby_features_trim = (
        groupby_features[list_variables]
        >> nvt.ops.ListSlice(0, maximum_length)
        >> nvt.ops.Rename(postfix="_trim")
    )

    # select the feature to return
    non_list_variable = [col + "-first" for col in session_feat]
    selected_features = groupby_features[non_list_variable] + groupby_features_trim

    workflow = nvt.Workflow(selected_features)
    dataset = nvt.Dataset(data, cpu=False)
    workflow.fit(dataset)
    session_data = workflow.transform(dataset).to_ddf().compute()

    # filter out small sessions
    list_variable = list_variables[0] + "_trim"
    if device == "cpu":
        session_data = session_data[session_data[list_variable].str.len() >= minimum_length]
    else:
        session_data = session_data[session_data[list_variable].list.len() >= minimum_length]

    return session_data


def save_time_based_splits(
    data,
    output_dir: str,
    partition_col: str = "day_idx",
    timestamp_col: str = "ts/first",
    test_size: float = 0.1,
    val_size: float = 0.1,
    overwrite: bool = True,
    cpu=False,
):
    """Split a dataset into time-based splits.
    Note, this function requires Rapids dependencies to be installed:
    cudf, cupy and dask_cudf

    Parameters
    -----
    data: Union[nvtabular.Dataset, dask_cudf.DataFrame]
        Dataset to split into time-based splits.
    output_dir: str
        Output path the save the time-based splits.
    partition_col: str
        Time-column to partition the data on.
    timestamp_col: str
        Timestamp column to use to sort each split.
    test_size: float
        Size of the test split, needs to be a number between 0.0 & 1.0.
    val_size: float
        Size of the validation split, needs to be a number between 0.0 & 1.0.
    overwrite: bool
        Whether or not to overwrite the output_dir if it already exists.
    cpu: bool, default False
        Whether or not to run the computation on the CPU.
    """

    if cpu:
        _save_time_based_splits_cpu(
            data,
            output_dir=output_dir,
            partition_col=partition_col,
            timestamp_col=timestamp_col,
            test_size=test_size,
            val_size=val_size,
            overwrite=overwrite,
        )

    return _save_time_based_splits_gpu(
        data,
        output_dir=output_dir,
        partition_col=partition_col,
        timestamp_col=timestamp_col,
        test_size=test_size,
        val_size=val_size,
        overwrite=overwrite,
    )


def _save_time_based_splits_gpu(
    data,
    output_dir: str,
    partition_col: str = "day_idx",
    timestamp_col: str = "ts/first",
    test_size: float = 0.1,
    val_size: float = 0.1,
    overwrite: bool = True,
):
    """Split a dataset into time-based splits.
    Note, this function requires Rapids dependencies to be installed:
    cudf, cupy and dask_cudf
    Parameters
    -----
    data: Union[nvtabular.Dataset, dask_cudf.DataFrame]
        Dataset to split into time-based splits.
    output_dir: str
        Output path the save the time-based splits.
    partition_col: str
        Time-column to partition the data on.
    timestamp_col: str
        Timestamp column to use to sort each split.
    test_size: float
        Size of the test split, needs to be a number between 0.0 & 1.0.
    val_size: float
        Size of the validation split, needs to be a number between 0.0 & 1.0.
    overwrite: bool
        Whether or not to overwrite the output_dir if it already exists.
    """

    try:
        import cudf
        import cupy
        import dask_cudf
        import nvtabular as nvt
    except ImportError:
        raise ValueError(
            "Rapids is necessary for this function, please install: "
            "cudf, cupy, dask_cudf & nvtabular."
        )

    if isinstance(data, dask_cudf.DataFrame):
        data = nvt.Dataset(data)
    if not isinstance(partition_col, list):
        partition_col = [partition_col]

    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    with tempfile.TemporaryDirectory() as tmpdirname:
        data.to_parquet(tmpdirname, partition_on=partition_col)
        time_dirs = [f for f in sorted(os.listdir(tmpdirname)) if f.startswith(partition_col[0])]
        for d in tqdm(time_dirs, desc="Creating time-based splits"):
            path = os.path.join(tmpdirname, d)
            df = cudf.read_parquet(path)
            df = df.sort_values(timestamp_col)

            split_name = d.replace(f"{partition_col[0]}=", "")
            out_dir = os.path.join(output_dir, split_name)
            os.makedirs(out_dir, exist_ok=True)

            cupy.random.seed(1)
            random_values = cupy.random.rand(len(df))
            train_size = 1 - val_size - test_size

            if train_size < 0:
                raise ValueError("train_size cannot be negative.")

            # Extracts 80% , 10%  and 10% for train, valid and test set, respectively.
            train_set = df[random_values <= train_size]
            train_set.to_parquet(os.path.join(out_dir, "train.parquet"))

            valid_set = df[
                (random_values > train_size) & (random_values <= (train_size + val_size))
            ]
            valid_set.to_parquet(os.path.join(out_dir, "valid.parquet"))

            test_set = df[random_values > (1 - test_size)]
            test_set.to_parquet(os.path.join(out_dir, "test.parquet"))


def _save_time_based_splits_cpu(
    data,
    output_dir: str,
    partition_col: str = "day_idx",
    timestamp_col: str = "ts/first",
    test_size: float = 0.1,
    val_size: float = 0.1,
    overwrite: bool = True,
):
    """Split a dataset into time-based splits.
    Note, this function requires Rapids dependencies to be installed:
    cudf, cupy and dask_cudf

    Parameters
    -----
    data: Union[nvtabular.Dataset, dask_cudf.DataFrame]
        Dataset to split into time-based splits.
    output_dir: str
        Output path the save the time-based splits.
    partition_col: str
        Time-column to partition the data on.
    timestamp_col: str
        Timestamp column to use to sort each split.
    test_size: float
        Size of the test split, needs to be a number between 0.0 & 1.0.
    val_size: float
        Size of the validation split, needs to be a number between 0.0 & 1.0.
    overwrite: bool
        Whether or not to overwrite the output_dir if it already exists.
    """

    try:
        import dask
        import numpy as np
        import nvtabular as nvt
        import pandas as pd
    except ImportError:
        raise ValueError(
            "Rapids is necessary for this function, please install: "
            "cudf, cupy, dask_cudf & nvtabular."
        )

    if isinstance(data, dask.DataFrame):
        data = nvt.Dataset(data)
    if not isinstance(partition_col, list):
        partition_col = [partition_col]

    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    with tempfile.TemporaryDirectory() as tmpdirname:
        data.to_parquet(tmpdirname, partition_on=partition_col)
        time_dirs = [f for f in sorted(os.listdir(tmpdirname)) if f.startswith(partition_col[0])]
        for d in tqdm(time_dirs, desc="Creating time-based splits"):
            path = os.path.join(tmpdirname, d)
            df = pd.read_parquet(path)
            df = df.sort_values(timestamp_col)

            split_name = d.replace(f"{partition_col[0]}=", "")
            out_dir = os.path.join(output_dir, split_name)
            os.makedirs(out_dir, exist_ok=True)

            np.random.seed(1)
            random_values = np.random.rand(len(df))
            train_size = 1 - val_size - test_size

            if train_size < 0:
                raise ValueError("train_size cannot be negative.")

            # Extracts 80% , 10%  and 10% for train, valid and test set, respectively.
            train_set = df[random_values <= train_size]
            train_set.to_parquet(os.path.join(out_dir, "train.parquet"))

            valid_set = df[
                (random_values > train_size) & (random_values <= (train_size + val_size))
            ]
            valid_set.to_parquet(os.path.join(out_dir, "valid.parquet"))

            test_set = df[random_values > (1 - test_size)]
            test_set.to_parquet(os.path.join(out_dir, "test.parquet"))
