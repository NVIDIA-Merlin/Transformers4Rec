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
from typing import Optional

from merlin.schema import Schema as CoreSchema
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata

from merlin_standard_lib import Schema


class Dataset:
    """Supports creating synthetic data for PyTorch and TensorFlow
    based on a provided schema.

    Parameters
    ----------
    schema_path : str
        Path to the schema file.
    """

    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        if self.schema_path.endswith(".pb") or self.schema_path.endswith(".pbtxt"):
            self._schema = Schema().from_proto_text(self.schema_path)
        else:
            self._schema = Schema().from_json(self.schema_path)

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def merlin_schema(self) -> CoreSchema:
        """Convert the schema from merlin-standardlib to merlin-core schema"""
        return TensorflowMetadata.from_json(self.schema.to_json()).to_merlin_schema()

    def torch_synthetic_data(
        self,
        num_rows: Optional[int] = 100,
        min_session_length: Optional[int] = 5,
        max_session_length: Optional[int] = 20,
        device: Optional[str] = None,
        ragged: Optional[bool] = False,
    ):
        """
        Generates a dictionary of synthetic tensors based on the schema.

        Parameters
        ----------
        num_rows : Optional[int]
            Number of rows,
            by default 100.
        min_session_length : int, optional
            Minimum session length,
            by default 5.
        max_session_length : int, optional
            Maximum session length,
            by default 20.
        device : torch.device, optional
            The device on which tensors should be created,
            by default None.
        ragged : bool, optional
            Whether sequence tensors should be represented with `__values` and `__offsets`,
            by default False.

        Returns
        -------
        Dict[torch.Tensor]
            Dictionary of tensors.
        """
        from transformers4rec.torch.utils import schema_utils

        return schema_utils.random_data_from_schema(
            self.schema,
            num_rows=num_rows,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
            device=device,
            ragged=ragged,
        )

    def tf_synthetic_data(self, num_rows=100, min_session_length=5, max_session_length=20):
        """
        Generates a dictionary of synthetic tensors based on the schema.

        Parameters
        ----------
        num_rows : Optional[int]
            Number of rows,
            by default 100.
        min_session_length : int, optional
            Minimum session length,
            by default 5.
        max_session_length : int, optional
            Maximum session length,
            by default 20.
        device : torch.device, optional
            The device on which tensors should be created,
            by default None.
        ragged : bool, optional
            Whether sequence tensors should be represented with `__values` and `__offsets`,
            by default False.

        Returns
        -------
        Dict[tf.Tensor]
            Dictionary of tensors.
        """
        from transformers4rec.tf.utils import schema_utils

        return schema_utils.random_data_from_schema(
            self.schema,
            num_rows=num_rows,
            min_session_length=min_session_length,
            max_session_length=max_session_length,
        )


class ParquetDataset(Dataset):
    """
    Class to read data from a Parquet file and load it as a Dataset.

    Parameters
    ----------
    dir : str
        Path to the directory containing the data and schema files.
    parquet_file_name : Optional[str]
        Name of the Parquet data file.
        By default "data.parquet".
    schema_file_name : Optional[str]
        Name of the JSON schema file.
        By default "schema.json".
    schema_path : Optional[str]
        Full path to the schema file.
        If None, it will be constructed  using `dir` and `schema_file_name`.
        By default None.
    """

    def __init__(
        self,
        dir,
        parquet_file_name: Optional[str] = "data.parquet",
        schema_file_name: Optional[str] = "schema.json",
        schema_path: Optional[str] = None,
    ):
        super(ParquetDataset, self).__init__(schema_path or os.path.join(dir, schema_file_name))
        self.path = os.path.join(dir, parquet_file_name)
