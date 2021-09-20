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

LOG = logging.getLogger("transformers4rec")


def download(output_path):
    """Download YooChoose dataset.

    Parameters
    ----------
    output_path: str
        Path to download the data to

    Returns output_path: str
    -------

    """
    from kaggle import api as kaggle_api

    kaggle_api.authenticate()

    LOG.info("Downloading data from Kaggle...")
    kaggle_api.dataset_download_files(
        "chadgostopp/recsys-challenge-2015", path=output_path, unzip=True
    )

    return output_path


def process_clicks(data_path: str, device="gpu"):
    """Process YooChoose dataset.

    Parameters
    ----------
    data_path: str
        Path to use to read the data from.
    device: str, default: "gpu"
        Device to use for processing clicks.

    Returns
    -------
    Union[cudf.DataFrame, pandas.DataFrame]
    """
    from .preprocessing import (  # type: ignore
        add_item_first_seen_col_to_df,
        remove_consecutive_interactions,
    )

    if device == "gpu":
        import cudf

        df = cudf.read_csv(
            data_path,
            sep=",",
            names=["session_id", "timestamp", "item_id", "category"],
            parse_dates=["timestamp"],
        )
    else:
        import pandas as pd

        df = pd.read_csv(
            data_path,
            sep=",",
            names=["session_id", "timestamp", "item_id", "category"],
            parse_dates=["timestamp"],
        )

    df = remove_consecutive_interactions(df)
    df = add_item_first_seen_col_to_df(df)

    return df
