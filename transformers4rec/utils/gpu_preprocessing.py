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
import shutil
import tempfile

from tqdm import tqdm


def save_time_based_splits(
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
            df.to_parquet(os.path.join(out_dir, "train.parquet"))

            cupy.random.seed(1)
            random_values = cupy.random.rand(len(df))

            # Extracts 10% for valid and test set.
            # Those sessions are also in the train set, but as evaluation
            # happens only for the subsequent day of training,
            # that is not an issue, and we can keep the train set larger.
            valid_set = df[random_values <= val_size]
            valid_set.to_parquet(os.path.join(out_dir, "valid.parquet"))

            test_set = df[random_values >= 1 - test_size]
            test_set.to_parquet(os.path.join(out_dir, "test.parquet"))
